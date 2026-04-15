# 第七章：collective 的语义：all-reduce、all-gather、reduce-scatter、all-to-all

> 如果只记住一句话，这一句就够了：collective 不是“通信库的黑话”，而是分布式系统里张量所有权如何变化的一套语义。

这一章最想做的事，不是把 NCCL 的实现细节背一遍，而是把通信先讲成一件朴素的事：某个张量原来在谁手里，现在要给谁看，最后到底谁还保留完整结果，谁只保留一部分。只要这个问题想清楚了，后面再看 all-reduce、all-gather、reduce-scatter、all-to-all，就不会觉得它们只是四个名字，而会觉得它们是在描述四种不同的“搬法”。

分布式训练里最容易混淆的，就是把“语义正确”和“性能正确”混在一起。前者问的是结果对不对，后者问的是这件事是不是跑得快、跑得稳、跑得划算。通信系统最怕的就是把这两件事倒过来：先纠结快不快，却没先弄明白张量到底该怎么流动。我们这一章先把语义稳住，再谈性能。

## 先从 4 卡纸面例子开始

想象四张卡，或者四个 rank。每张卡上先各有一块自己的张量分片。它们可以是参数分片、梯度分片、激活分片，也可以是路由后的一小段 token。先不要急着问底层走了哪条链路，先问最简单的问题：这四张卡最终要得到什么。

如果最终每张卡都要得到“所有分片加起来后的同一个完整结果”，那就是 all-reduce。比如四张卡分别算出一份梯度，训练时要把它们求和或者求平均，然后每张卡都拿到同一份完整梯度继续更新参数。这里的关键不是“传了多少包”，而是“结果的拥有权”没有变：每张卡最后都拿到了完整结果。

如果最终每张卡都只想得到“别人的一部分也拼进来后的完整张量”，那就是 all-gather。它和 all-reduce 很像，都是把分散的信息拼完整，只不过 all-gather 不做归约，只做收集和广播。你可以把它理解成：每张卡原来手里只有一页，最后大家都要读到整本书。

如果相反，目标不是让大家都拿完整结果，而是先做归约，再按 rank 切回去，那就是 reduce-scatter。它的结果刚好和 all-reduce 相反：先把四份信息合成一个整体，再把这个整体切成四块，每张卡只保留自己那一块。对系统来说，这一步很重要，因为很多时候我们并不真的需要把完整结果长期放在每张卡上；我们只需要“合完再分”，这样就能少占一些显存，也能少搬一些后续要再切开的数据。

从语义上看，reduce-scatter 后接 all-gather，可以重新拼回和 all-reduce 等价的完整结果。NCCL 的 collective 文档就明确把这条等价关系写了出来，这很有用，因为它提醒我们：有时系统真正想要的不是“每一步都保留完整态”，而只是“在某个时刻暂时变完整，再重新切回分片态”。([NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html))

最后是 all-to-all。这个名字听起来最像通信，而它的语义也最“交换”：每张卡都要把自己的不同部分发给其他每一张卡，同时也从其他每一张卡收回不同部分。这里不再是“大家都拿同一份结果”，而是“每个人都拿到不同的那几块”。这类语义最常出现在路由、重排、专家分发这些问题里，因为数据的归属权本来就不是固定的。

> [!abstract] 定义：rank、shard、replica、owner
> rank 是并行中的一个参与者。
> shard 是张量的一段分片。
> replica 是同一份完整内容的副本。
> owner 是当前真正持有并负责使用这份数据的一方。
> 
> 这一章所有 collective，都是在改写 owner 的边界。

## DP、TP、CP 到底在切什么

很多人第一次接触分布式训练时，会觉得 Data Parallel、Tensor Parallel、Context Parallel 这些缩写像是三种不同流派。其实它们没有那么神秘。更朴素的问法是：**你到底把什么切开了。**

Data Parallel 切的是**样本**。每个 rank 都保留一份完整模型副本，参数是 replica，前向和反向各自处理不同 mini-batch 分片。这样做的好处是，单卡上每一步看到的计算图很完整，坏处是参数和优化器状态在每张卡上都要复制一份。它最核心的通信通常发生在 backward 之后：每张卡都算出自己的梯度，再把这些梯度 all-reduce 成一致。换句话说，DP 的关键词不是“模型被切开”，而是“数据被切开，参数保持副本态”。

Tensor Parallel 切的是**层内部的张量维度**。这里最值得用线性层来理解。设某层是 `y = xW`。如果把 `W` 沿着输出维切开，那么每个 rank 只持有一部分列，各自算出 `y` 的一部分输出块；这样得到的 `y` 天然是分片态，后面是否要 all-gather，要看下一步是不是非要完整结果。如果反过来，把 `W` 沿着输入维切开，那么每个 rank 只能算出一份局部 partial result，最后必须把这些 partial 合起来，常见做法就是 all-reduce 或 reduce-scatter。于是 TP 的本质不是“多卡一起算更大的模型”这么抽象，而是：**层内部哪些维度被切开，决定了这一步结束后张量应当停在 shard 态，还是必须先回到完整态。**

Context Parallel 切的是**序列 / 上下文维度**。它和 DP 最大的区别在于：DP 里不同 rank 处理的是不同样本，彼此天然独立；而 CP 里，不同 rank 往往处理的是同一个样本的不同 token 段，这些 token 之间在 attention 里又并不独立。也就是说，CP 不是简单把 batch 维换成 sequence 维；它真正困难的地方在于，某个 rank 虽然只持有一段上下文，但本地 query 仍然可能需要别的 rank 上的 key / value。于是 CP 天然会牵扯到 all-gather、all-to-all 或 ring 式传递 K/V 这类动作。它切开的不是参数，也不只是样本，而是**同一个样本内部的上下文拥有权**。

如果把这三者压缩成一句非常短的对照，那就是：

> [!abstract] 一个不神秘的区分
> DP：切样本，参数保留副本态。  
> TP：切层内张量维度，参数和激活常常一起进入分片态。  
> CP：切序列维，激活和上下文状态沿 token 维分片，但 attention 依赖会迫使它们再次交换。

这样一来，collective 就不再像“通信章节的专有词汇”，而像并行策略的语法。DP 之所以离不开 all-reduce，是因为梯度最后要重新一致；TP 之所以频繁遇到 all-gather 和 reduce-scatter，是因为分片态和完整态要在层与层之间来回切换；CP 之所以让 all-gather、all-to-all 或 ring 交换变重要，是因为同一段长上下文根本不是本地就能独立算完的。

## 语义先于实现

把语义讲清楚之后，很多训练策略就会自然地浮出来。DDP 是最直观的：每张卡都跑同一个模型副本，各自算出梯度，再用 all-reduce 把梯度同步成一致。这里的目标非常明确，就是让所有 replica 在更新时看到同一份结果。语义上它是“大家最终一致”；实现上才去考虑怎么把这一致性做得更快。

Tensor Parallel 往往更像 reduce-scatter 和 all-gather 的组合。因为模型的某些矩阵乘法或激活不适合整块复制，只能沿着某个维度切开。切开之后，有些地方要先聚合再分发，有些地方要先拿到别人分来的块再继续算。于是 TP 的问题就不再是“我这里有没有完整张量”，而是“这一步算完以后，张量应该停留在分片态还是完整态”。

Context Parallel 也是类似的思路，只不过它常常切的是序列维。长上下文不一定非要完整地压在一张卡上，激活可以沿着上下文切开，然后在需要整合信息的地方再做收集或交换。这样一来，collective 不再只是“通信操作”，而是模型如何跨序列组织状态的一部分。

MoE 和专家并行则更能说明 all-to-all 的必要性。路由器一旦决定某个 token 应该去哪个 expert，数据就不再是“每张卡都拿同一份”，而是“不同 token 去不同地方”。token 要分发，专家处理完又要把结果收回来，这就天然是 all-to-all 的语义。很多人第一次看 MoE 觉得它像是“多了几层网络”，其实它更像是“把张量的归属权重新分配了一遍”。

## 为什么 reduce-scatter 和 all-gather 很重要

如果只看名字，reduce-scatter 似乎只是 all-reduce 的一半，all-gather 似乎只是把碎片拼起来。但从系统角度看，这两步很重要，因为它们给了我们一个更灵活的思路：不是所有结果都非得在每一张卡上完整出现。

这件事对显存特别敏感。很多时候，系统不是算不动，而是放不下。把完整结果长期复制在每张卡上，显存会越来越紧；把结果维持在分片态，再在真正需要的时候聚合或展开，系统就更容易活得下来。换句话说，collective 不只是“把数传过去”，它还在决定状态该以什么粒度存在。

这也是为什么很多并行设计本质上都是在选“在哪一步保留完整态，在哪一步保留分片态”。如果把这个问题看清楚，DDP、TP、CP、EP 之间的差别就不会显得那么神秘。它们当然有不同的计算图、不同的负载形态、不同的瓶颈，但底层都绕不开同一个问题：状态该在哪里完整，在哪里切开，在哪里再合起来。

## 现实插页：研究者真的会写到的 collective 代码

如果你在真实训练代码里看 distributed API，最常见的往往不是某个巨大框架的内部实现，而是几行非常直接的 `torch.distributed` 调用。把这些 API 看熟，比先背 ring / tree 更有帮助。下面这段代码不是完整训练脚本，而是一个最小的 collective 语义速写：

```python
import torch
import torch.distributed as dist


# init_process_group 会把所有 rank 连成一个通信组。
# backend="nccl" 是 CUDA 多卡训练里最常见的后端。
dist.init_process_group(backend="nccl")

# rank 是“我是谁”，world_size 是“总共有多少参与者”。
rank = dist.get_rank()
world_size = dist.get_world_size()


# ---- all-reduce：每个 rank 最后都拿到同一个归约结果 ----
grad = torch.ones(1024, device="cuda") * (rank + 1)
dist.all_reduce(grad, op=dist.ReduceOp.SUM)


# ---- all-gather：每个 rank 原来只有自己的 shard，最后都拿到完整拼接结果 ----
local_shard = torch.full((256,), rank, device="cuda")
full_tensor = torch.empty(256 * world_size, device="cuda")
dist.all_gather_into_tensor(full_tensor, local_shard)


# ---- reduce-scatter：先归约，再让每个 rank 只保留其中一段 ----
# 输入张量通常可以理解成“本 rank 提供的一份完整 partial”。
partials = torch.ones(256 * world_size, device="cuda") * (rank + 1)
reduced_shard = torch.empty(256, device="cuda")
dist.reduce_scatter_tensor(reduced_shard, partials, op=dist.ReduceOp.SUM)


# ---- all-to-all：每个 rank 都把不同块发给不同 rank ----
# 这里 send 的第 0 维可以先理解成“我要发给各个目标 rank 的分块”。
send = torch.arange(world_size * 4, device="cuda").reshape(world_size, 4) + rank * 100
recv = torch.empty_like(send)
dist.all_to_all_single(recv, send)


# synchronize 让 host 等到前面发出去的 CUDA / NCCL 工作都真正完成，
# 这样调试和 profile 时才不会把异步执行的尾巴混在后面。
torch.cuda.synchronize()
```

这段代码有一个很好的教学价值：它把 collective 从抽象名词变成了你真的会在项目里看到的 API。`all_reduce` 对应“最后大家一致”；`all_gather_into_tensor` 对应“大家都把完整结果拼回来”；`reduce_scatter_tensor` 对应“先合再分”；`all_to_all_single` 对应“每个人都和每个人交换不同块”。如果以后你再去看 DDP、TP、CP、MoE 路由的真实实现，就会更容易看出：这些高层策略底下，最后也还是在反复调用这几类语义。

## 性能问题从哪里来

语义讲清楚以后，才轮到性能。性能不是和语义分开的另一件事，而是语义在现实硬件上的代价。一个 collective 是否快，取决于数据大小、节点数、链路带宽、拓扑、实现方式，以及它是否能和别的工作重叠。

先说最直观的一点：同样叫 all-reduce，张量很小的时候，开销可能主要来自 launch 和同步；张量很大的时候，开销可能主要来自带宽和链路利用率。前者更像“动作太碎”，后者更像“搬得太多”。所以我们不能只问“用了什么 collective”，还要问“这个 collective 传的是什么尺寸的东西”。

还有一条很基础、但在工程里极容易踩坑的约束：collective 的参与者必须对这次通信有同样的理解。NCCL 文档明确要求，每个 rank 都要用同样的 count 和 datatype 参与同一个 collective；如果不满足，后果就不再只是“慢一点”，而可能是直接 hang 住、crash，或者把错误的数据一路传下去。语义如果没有先对齐，性能分析根本无从谈起。([NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html))

再往下看，拓扑也会影响结果。四张卡如果在同一台机器里，和四张卡分布在两台机器上，通信味道就会不一样。ring、tree 这些名字，本质上是在说：数据按什么顺序流，经过几跳，是否更容易把带宽吃满，是否更容易把延迟压低。你不需要在这一章把它们展开成算法课，但需要知道它们不是装饰性的实现名词，而是直接影响 wall-clock 的路径选择。

> [!tip] 直觉：同一个 collective，不同实现的气质不一样
> ring 常常更像“把大块数据沿着环慢慢推过去”，适合把带宽利用起来。
> tree 常常更像“先分后合”，在某些规模下可以更快地缩短延迟。
> topology 则决定了这些路径到底有没有机会成立。

因此，collective 的性能问题有两层。第一层是它有没有选择正确的语义，也就是张量该怎么流。第二层是它有没有找到合适的流法，也就是数据该按什么顺序、沿什么拓扑、在什么时候和别的计算重叠。我们做系统分析时，必须先守住第一层，再进入第二层。否则很容易在“优化通信”这件事上忙得很热闹，却连张量到底该不该完整落在每张卡上都没想明白。

## 结尾：先认清语义，再谈速度

如果把这一章压缩成一句总结，那就是：collective 的第一任务不是让网络忙起来，而是让张量在分布式系统里以正确的方式换主人。

all-reduce 解决的是“大家最后要一致”；all-gather 解决的是“大家都要看见完整内容”；reduce-scatter 解决的是“先合再分，别把完整结果长期摊在每张卡上”；all-to-all 解决的是“不同数据要去不同地方”。这些语义一旦稳定下来，DDP、TP、CP、EP、MoE 路由就不再是零散缩写，而是同一套状态流动语言里的不同句法。

等你真的开始看 profiler，或者看分布式训练的 trace，你会发现通信问题通常不是一团雾。它更像几类很具体的等待：有人在等别人发完，有人在等自己先聚完再继续，有人在等路由后的 token 回来，有人在等链路把这批分片推完。到了那时，你就会知道，collective 不是“网络那部分的黑箱”，而是分布式训练里最清楚、也最容易被误读的一层语义。

语义正确，是第一步；性能正确，是第二步。把这两步分开想，很多通信问题就不会再显得模糊了。

## 代码练习
- 练习 1：[[AI Infra/硬件、运行时与 profiling/Chap7/scripts/collective_semantics.py|collective_semantics.py]]；答案 [[AI Infra/硬件、运行时与 profiling/Chap7/scripts/collective_semantics_answer.py|collective_semantics_answer.py]]
- 练习 2：[[AI Infra/硬件、运行时与 profiling/Chap7/scripts/collective_cost_toy.py|collective_cost_toy.py]]；答案 [[AI Infra/硬件、运行时与 profiling/Chap7/scripts/collective_cost_toy_answer.py|collective_cost_toy_answer.py]]
- 索引：[[AI Infra/硬件、运行时与 profiling/Chap7/scripts/README]]

## 思考题答案

### 练习 1：collective semantics

这道题的关键不是把 Python list 玩熟，而是把“谁最终拥有完整结果、谁最终只保留一部分”这件事看熟。一旦这个所有权变化清楚了，collective 的名字就不会再显得像黑话。

**1）为什么 reduce-scatter + all-gather 可以重新组成 all-reduce？**

因为 all-reduce 的语义，本质上就是“两步合在一起”：先把所有 rank 的 partial result 做归约，再让每个 rank 都拿到完整结果。reduce-scatter 做的是前半句——它先把结果归约好，但只把归约后的一个 shard 留在每个 rank 上；all-gather 做的是后半句——把这些 shard 再拼回完整张量，并让所有 rank 都看见同一份完整结果。所以从语义上说，reduce-scatter 后接 all-gather，正好重新组成一次 all-reduce。

**2）DDP 更像哪一种？MoE token 路由更像哪一种？**

DDP 最像 all-reduce。因为它的目标很明确：每个 rank 本地算出梯度之后，大家必须重新看到同一份完整梯度，才能保持参数副本一致。MoE token 路由则更像 all-to-all。因为每个 token 最终去哪个 expert，并不是“大家都拿同一份结果”，而是“不同 token 被送到不同地方，再把结果收回来”。它的本质是数据拥有者和消费者一起变化。

**3）哪些 collective 会让“完整张量”停留在每个 rank 上？**

最典型的是 all-reduce 和 all-gather。all-reduce 之后，每个 rank 都得到同一个归约后的完整结果；all-gather 之后，每个 rank 也都得到拼接好的完整张量。reduce-scatter 则相反，它故意让完整结果不要长期停在每个 rank 上，而是回到分片态；all-to-all 更不是“每个人都拿完整结果”，而是“每个人都拿到属于自己的那些不同块”。

### 练习 2：collective cost toy

这道题的作用，是把前一题的语义继续推进到“每个 rank 大概要搬多少字节”这个层面。它还不是完整的通信性能模型，但足够建立一个很重要的工程直觉：**语义不同，搬运成本的形状也不同。**

**1）为什么 all-reduce 常常比 all-gather 更贵？**

在这个 toy 记账里，all-gather 可以理解成“把别人那几份 shard 拼过来”，而 ring all-reduce 则等价于 reduce-scatter 加 all-gather 两步。也就是说，all-reduce 不只是“把东西收齐”，它还先做了一轮归约再分发，所以每个 rank 的总搬运量通常会比单独的 all-gather 更高。从这个角度看，all-reduce 更贵，并不神秘，它只是比 all-gather 多做了一层“先合”的动作。

**2）为什么 reduce-scatter + all-gather 是一个重要组合？**

因为它让系统可以更灵活地控制“完整态到底出现多久”。有些时候，我们并不需要完整结果一直停在每张卡上，只是需要在某个时刻短暂完成归约，然后很快再切回分片态继续算。这样显存占用往往更可控，也更符合 TP、ZeRO 一类系统的状态管理方式。reduce-scatter + all-gather 的重要性，恰恰就在于它把“先合再分”和“需要时再拼”拆成了两个独立步骤。

**3）为什么 all-to-all 往往更容易暴露拓扑和负载不均的问题？**

因为 all-to-all 不是“大家都拿同一份结果”，而是“每个 rank 都要和每个其他 rank 交换不同块”。这会让通信路径更分散，也更容易出现局部热点。如果某些 rank 发得特别多、某些 expert 特别忙、某些链路特别挤，那么 all-to-all 往往会比 all-reduce 更早暴露出负载不均和拓扑限制。换句话说，它不只是总字节数的问题，更是“谁在和谁交换、交换得均不均匀”的问题。

## 本章 API 速记

- `torch.nn.parallel.DistributedDataParallel (DDP)`：最常见的数据并行入口；模块参数保留副本态，梯度在 backward 过程中同步。
- `torch.distributed.tensor.parallel.parallelize_module(...)`：PyTorch 当前 TP API 的常见入口，用给定 parallelize plan 把模块改造成 tensor parallel 版本。
- `torch.distributed.tensor.experimental.context_parallel(...)`：PyTorch 当前 CP 教程里使用的上下文并行入口；可以把序列维上的 buffer / activation 按上下文分片处理。
- `torch.distributed.init_process_group(...)`：初始化分布式通信组，让多个 rank 进入同一套 collective 语义。
- `dist.get_rank()`：返回当前进程 / 当前参与者的 rank 编号，也就是“我是谁”。
- `dist.get_world_size()`：返回当前通信组里总共有多少个 rank 参与。
- `dist.all_reduce(tensor, op=...)`：对所有 rank 的同名张量做归约，并让每个 rank 都拿到同一份结果。
- `dist.all_gather_into_tensor(output, input)`：把每个 rank 的本地 shard 收集并拼接成完整张量，让每个 rank 都拿到这份完整结果。
- `dist.reduce_scatter_tensor(output, input, op=...)`：先归约，再把结果按 rank 切分；每个 rank 最后只保留其中一个 shard。
- `dist.all_to_all_single(output, input, ...)`：把输入张量按块拆开，分别发给不同 rank，再把收到的块拼回输出。
- `dist.ReduceOp.SUM`：最常见的归约方式之一，表示把各个 rank 的值相加。
- `torch.cuda.synchronize()`：等待当前 CUDA / NCCL 工作真正完成；调试、计时和 profiler 时常用来隔开异步执行。

## 本章引用与延伸阅读
- [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [NVIDIA Collective Communication Library (NCCL) Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [Distributed communication package — torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [DistributedDataParallel — PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [Tensor Parallelism — PyTorch documentation](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel)
- [torch.distributed.tensor — PyTorch documentation](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
