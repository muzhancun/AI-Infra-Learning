# 第六章：kernel、stream 与 launch overhead

> GPU 往往不是慢在“算”，而是慢在“喂”。

如果把一个 PyTorch 里的 op 放到时间线上看，它很少还是我们在代码里写下的那个名字。真正进入 GPU 的，通常是一串 kernel；真正决定体感速度的，也往往不是单个 kernel 的峰值性能，而是这些 kernel 之间有没有空隙、有没有被拆碎、有没有因为形状变化而重新编译，最后有没有把 host 端和 device 端都拖慢。

这一章要讲的，就是这条 timeline。launch 是什么，为什么 launch 本身会变成成本；stream 为什么只是一个有序队列，而不是自动并行按钮；小 kernel 为什么特别容易把时间浪费在缝隙里；`torch.compile` 到底怎样改变 runtime；以及什么时候 CUDA Graphs 能把这些缝隙收紧，什么时候它又帮不上忙。

## 什么叫 launch，为什么它会成为成本

在 CUDA 的语义里，kernel launch 是一个异步提交动作：host 端把工作送出去，函数调用本身很快返回，真正的执行稍后才在 GPU 上发生。CUDA Programming Guide 把 stream 和异步执行说得很清楚：kernel launch、内存拷贝、事件同步，都是在“提交”和“完成”之间留出时间差的机制。([CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/asynchronous-execution.html))

这件事放到深度学习里，就意味着一次 op 并不是“调用一下就算完”。Python 先要走一遍函数栈，框架要做 dispatcher 和参数打包，driver 要提交命令，随后 GPU 才开始真正执行。只要某个 kernel 足够大，这些开销就会被淹没；可一旦 kernel 变小，launch overhead 就会从背景噪音变成显眼成本。

这也是为什么很多看起来“很快”的小 op，在 trace 里并不快。它们不是算得慢，而是前后准备和提交的成本占了太多比例。你看到的不是一个长长的计算块，而是一串短小的执行段，中间夹着 host 端的空白、同步点和调度等待。对这种工作负载来说，优化的第一步往往不是换一个更快的算子，而是先让它少发几次、少拆几段、少来回跳。

## stream 是顺序队列，不是自动并行按钮

很多人第一次看 CUDA stream 时，会直觉地把它理解成“多开几条流水线就会更并行”。这理解只对了一半。stream 的本质是有序队列：同一条 stream 里的操作不会随便跳序，前一个没完，后一个就不会越过去。CUDA 官方文档明确说明，stream 里的操作按顺序执行，所谓异步，是指提交返回得快，不是指执行顺序混乱。([CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/asynchronous-execution.html))

真正能不能并行，要看两个层面。第一，是不是有依赖关系：后一个 kernel 是否真的要等前一个的结果。第二，是硬件和资源是否允许：即使没有依赖，也不等于一定能并发，寄存器、shared memory、block 配置、调度器和内存带宽都会限制重叠的程度。默认 stream 还会带来额外的同步语义，容易把本来可能并行的工作串起来。换句话说，stream 不是“并行开关”，它只是让你更精确地表达依赖。

把这个心智模型放到 trace 里，就会很有用。若你看到 kernel 之间明明没有逻辑依赖，却还是一段接一段地排着队，问题可能不在 GPU 算得慢，而在提交路径、默认 stream 语义，或者 host 没把足够多的独立工作提前喂进去。很多所谓“GPU 利用率不高”，其实只是队列里没有足够多的活。

## 小 kernel 为什么最容易把时间浪费在缝隙里

小 kernel 的麻烦，在于它会把“执行”这件事切得太碎。单个 kernel 的计算量不大，launch 次数却很多，于是时间线就会出现一种很典型的形状：一个个短促的执行块之间，夹着不成比例的缝隙。缝隙里可能是 host 端在准备下一个 kernel，也可能是 Python 在跑控制流，也可能是 runtime 在等前一个操作完成后才能继续。结果就是，GPU 看起来一直在忙，但忙得不连续。

`relu`、`layernorm` 这类操作尤其容易暴露这个问题。它们本身不一定复杂，可每次做的事情都不多，单独发射时很难把 launch 成本摊薄。相比之下，大矩阵乘法的执行块更长，launch overhead 相对不显眼。于是同样在做“算子优化”，真正值得优化的对象未必是最“重”的那个，而可能是最“碎”的那个。

这也是为什么 `torch.compile` 会被放进这一章，而不是只放在第二部的训练编译章节里。它最直接改变的，不是抽象语法，而是执行形状。编译器尽量把多个小操作拼成更大的图，减少 kernel 数量，压缩提交次数，把原本散落的执行段收拢成更连贯的块。PyTorch 官方文档把 `torch.compile` 描述为图捕获与加速的入口，并明确点出了 TorchDynamo、AOTAutograd 和 TorchInductor 这些名字；但对这一章来说，更重要的是它们如何改变 timeline，而不是它们各自内部怎么实现。([torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html), [torch.compiler](https://docs.pytorch.org/docs/stable/torch.compiler.html))

## `torch.compile` 改变的是 runtime，而不只是代码形态

`torch.compile` 最容易让人误会的一点，是把它理解成“把 Python 代码翻译一下”。这不够。它真正做的，是尝试把可追踪的部分收进一个图里，然后交给后端去做融合、调度和生成。只要图连续，runtime 就有机会减少 launch、融合小算子、复用一些中间结果；一旦图被打断，这些机会就会被切碎。

这里最关键的词是 graph break。PyTorch 的 troubleshooting 文档把它定义得很直接：当代码不能被追踪时，编译器会在这里断开，先编译已经拿到的部分，执行那段不支持的 Python，再从后面继续追踪。问题不只是“少编译了一点”，而是原本连在一起的优化窗口被切碎了。图一旦断开，编译器能看到的上下文就少了，fusion 的空间变小，调度也更难统一。([Working with Graph Breaks](https://docs.pytorch.org/docs/stable/compile/programming_model.graph_breaks_index.html))

形状变化会让事情更复杂。动态 shape 不是坏东西，很多真实 workload 本来就有可变长度、可变 batch、可变分支；但 shape 一旦不稳定，编译器就要在静态假设和动态放宽之间做选择。假设越静态，生成代码越锐利，复用也越容易；假设越动态，适应性越强，但有些优化空间会变窄。更糟的是，如果形状变化频繁到超出当前图的适配范围，就会触发重新编译。于是 timeline 上除了正常的执行段，还会多出一类新的成本：第一次编译慢、某次 shape 变化后又慢、图断了之后又慢。PyTorch 的文档也明确提到，自动动态 shape 会在形状变化时尝试放宽假设，但这并不意味着编译成本会凭空消失。([Dynamic Shapes](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html), [torch.compiler.config](https://docs.pytorch.org/docs/stable/torch.compiler.config.html))

> [!tip] 练习里说的 compile cache miss，可以先用一个很朴素的心智模型理解
> 先把编译器想成这样一台机器：它会为“某一类输入形状”生成一份已经编好的版本，并把这份版本放进 cache 里。下次如果又来了“同一类形状”，就可以直接复用；如果来了 cache 里没见过的新形状，就要重新编。
> 
> 最保守的做法，是把 **exact shape** 当成 cache key。这样专门化程度高，但只要形状轻微抖动，就可能产生新的 key。更宽松的做法，则是允许一批“相近形状”共用一个 key，例如把 sequence length 按 bucket 归类。这样重编译次数可能减少，但生成代码也往往不如对单一静态 shape 那么锐利。
> 
> 真实 `torch.compile` 的 dynamic shapes 比这个复杂得多，它不是简单地“按桶四舍五入”，而会结合 guards、符号化 shape 和可接受的动态范围来决定复用与重编译。但用这个 toy 心智模型，已经足够帮助我们理解：**shape 抖动为什么会出现在 timeline 上，而且常常表现成一阵阵额外的慢。**

> [!tip] 看 `torch.compile` 时最值得盯的，不是“有没有用上”，而是 timeline 有没有变得更连续
> 如果原来很多短 kernel 被收拢成少数几个更长的执行段，通常是好信号。
> 如果 trace 里出现了编译热身、重编译、graph break 反复出现，那说明 runtime 并没有真正稳住。

## CUDA Graphs 适合什么，不能解决什么

CUDA Graphs 的价值也很朴素：把一段相对稳定的工作先捕获下来，之后反复 replay。PyTorch 官方文档把 `torch.cuda.graph` 和 `torch.cuda.CUDAGraph` 描述成一种捕获 GPU 工作、再以相同工作流重放的机制；它的主要收益是减少 Python、C++ 和 driver 这些提交路径上的开销。只要工作足够稳定，尤其是形状和控制流比较固定，Graph replay 就很适合用来压缩 launch overhead。([torch.cuda.graph](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html), [torch.cuda.CUDAGraph](https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html))

但 CUDA Graphs 不是万能药。它要求工作足够 graph-safe，通常意味着静态 shape、静态控制流，至少还要能保证 replay 时的内存地址和依赖关系稳定。换句话说，它擅长的是“把重复工作做得更省”，不擅长的是“把不稳定的工作变稳定”。如果你的瓶颈是 memory-bound、算子本来就太慢，或者输入形状变化过于频繁，那么 Graphs 的收益就会有限。

所以这里应该形成的判断不是“能不能上 CUDA Graphs”，而是“我的问题是不是 launch 和 CPU overhead 主导的重复路径”。如果是，小而重复、结构稳定的那段工作，确实值得考虑捕获；如果不是，先去看图是否被 graph break 切碎、shape 是否太抖、stream 是否把工作串死，通常更有效。

## 现实插页：同一段 FLA 代码在 eager、compile 和 varlen 下会怎样出现

到了真正的研究代码里，`torch.compile` 很少是单独出现的。更常见的情况是：你拿一层真实模块，加上 profiler，看 steady-state 和 shape 抖动时 timeline 到底变成什么样。下面这段代码把 FLA README 里的 layer 用法，和 PyTorch 官方推荐的 warm-up + profiler 工作流拼在一起：

```python
import torch
from fla.layers import MultiScaleRetention

device, dtype = "cuda", torch.bfloat16
hidden_size, num_heads = 1024, 4

layer = MultiScaleRetention(
    hidden_size=hidden_size,
    num_heads=num_heads,
).to(device=device, dtype=dtype)

# torch.compile 会尝试把可追踪的计算收进图里，再交给后端做融合与代码生成。
# 这里先把 eager 模块包成 compiled 版本，后面实际调用 layer_c(x)。
layer_c = torch.compile(layer)


def run(seqlen: int):
    # 这里故意把 sequence length 暴露成变量，方便之后观察 shape 抖动的影响。
    x = torch.randn(8, seqlen, hidden_size, device=device, dtype=dtype)
    y, *_ = layer_c(x)
    return y


# 先用稳定 shape 做一次 warm-up，把首次 compile 和 lazy init 提前发生掉
run(2048)

# synchronize 的目的还是一样：让 warm-up 阶段真的结束，再开始正式 profile。
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # 记 shape 很关键，因为这一节正想观察 shape 变化是否带来新的 runtime 行为。
    record_shapes=True,
) as prof:
    # 前两个 2048 可以看 steady-state；
    # 中间插入 2176 是为了制造一次轻微 shape 抖动。
    for seqlen in [2048, 2048, 2176, 2048]:
        run(seqlen)
    torch.cuda.synchronize()

# 这里按 CPU 自身时间排序，更容易先看到 compile / host-side 开销。
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

# 导出 trace 之后，就可以去时间线里找：
# 1) kernel 是否更连续了
# 2) shape 抖动后 CPU 侧是否重新长出缝隙
prof.export_chrome_trace("fla_compile_trace.json")
```

这段代码最适合观察三件事。第一，稳定 shape 下，compile 之后的 timeline 有没有比 eager 更连续、kernel 数量有没有变少。第二，sequence length 轻微抖动时，CPU 侧有没有重新长出空隙，trace 里有没有更多编译、graph break 或 runtime 波动的迹象。第三，GPU kernel 本身的时间占比和 CPU 提交时间占比，是否在不同 shape 下开始重新洗牌。`flame` 的 README 里同时出现了 `--training.compile` 和 `--training.varlen` 这两个 flag，这也正是为什么本章第二个 toy 题在模拟 shape jitter：在真实训练代码里，这不是抽象问题，而是很容易真的出现在时间线上。

当然，真实项目里并不保证 compile 一定收益稳定。`flame` 也明确提醒，`fla` 里有不少 fused kernels，可能和 `torch.compile` 存在交互问题。因此更稳妥的做法从来不是“开 compile 看运气”，而是**先用 profiler 看证据，再判断 compile 到底是在收紧 timeline，还是在引入新的缝隙。**

## 结尾：先学会看 timeline 的缝

这章最想留下的不是一个工具清单，而是一种看法。下次你打开 `torch.profiler`、Nsight Systems，或者任何能看时间线的工具时，不要先盯“哪个 kernel 最慢”，而是先看缝隙在哪里：kernel 前的等待，kernel 之间的空白，graph break 后的断裂，shape 变化后的重编译，stream 之间被意外串起来的段落。

如果一段工作在代码里看起来很平滑，但在 trace 里显得支离破碎，那大概率就不是“模型突然变笨了”，而是 runtime 没把它连续地送上 GPU。反过来，如果你能把这些缝一点点缝合起来，很多原本说不清的“GPU 不够快”，就会变成可以下手的具体问题。

下一章我们会继续往下走，把这种“看时间线找问题”的方法带到多卡通信上。因为一旦从单卡走向多卡，timeline 上的缝隙就不再只是 launch、host gap 和 shape 抖动，它还会变成 rank 之间的等待、同步和拖尾。

## 代码练习
- 练习 1：[[AI Infra/硬件、运行时与 profiling/Chap6/scripts/timeline_gaps.py|timeline_gaps.py]]；答案 [[AI Infra/硬件、运行时与 profiling/Chap6/scripts/timeline_gaps_answer.py|timeline_gaps_answer.py]]  
  在一个 toy timeline 里比较 eager、fusion 和 graph replay，练习区分“真正的 kernel 时间”和“launch / host 造成的缝隙”。
- 练习 2：[[AI Infra/硬件、运行时与 profiling/Chap6/scripts/compile_cache_toy.py|compile_cache_toy.py]]；答案 [[AI Infra/硬件、运行时与 profiling/Chap6/scripts/compile_cache_toy_answer.py|compile_cache_toy_answer.py]]  
  在一个 variable-shape toy 场景里观察 exact key 和 bucketed key 的 cache hit / miss，练习理解 shape 抖动为什么会变成重编译。
- 索引：[[AI Infra/硬件、运行时与 profiling/Chap6/scripts/README]]

## 思考题答案

### 练习 1：timeline gaps

这道题的重点，不是比较哪一个数字更漂亮，而是把总时间拆开看：到底多少是真正在 GPU 上做 kernel 的时间，多少是 launch、host 提交和 runtime 缝隙造成的时间。参考答案里，`kernel-only` 是 153 us；而 eager 总时间是 193 us，这意味着有 40 us 都不是“纯 kernel 计算”，而是额外的提交与缝隙成本。fused 和 graph replay 的意义，正是在于把这部分缝隙压小。

**1）为什么最短的小 kernel 最容易被 launch overhead 吃掉？**

因为 launch overhead 更像一种近似固定成本，而不是随 kernel 数学复杂度线性缩放的成本。对一个只跑几微秒的小 kernel 来说，前面的 Python、dispatcher、runtime、driver 提交这些动作，可能和它真正执行的时间差不多，甚至更长。这样一来，你在 timeline 上看到的就不再是“算得慢”，而是“每次真正开始算之前，都先花了一段不成比例的准备时间”。这也是为什么小 kernel 特别容易把时间浪费在缝里。

**2）这里哪种优化更像解决 kernel 过碎，哪种更像进一步压缩 Python/host 提交开销？**

在这个 toy 里，**fusion** 更像是在解决 “kernel 过碎”。它把几个本来各自 launch 的小操作合在一起，让原本多段短执行和多次 launch 变成更少的段落，所以最直接减少的是“碎片化”。而 **graph replay** 更像是在此基础上，继续压缩 Python / host / driver 这条提交路径上的重复成本。也就是说，fusion 更像先把活拼大，graph replay 更像把反复提交这件事本身做得更省。

**3）如果 shape 频繁变化，这类收益为什么会更不稳定？**

因为无论是 fusion 还是 graph replay，都更依赖“工作形状足够稳定”这个前提。shape 一旦频繁变化，编译器可能需要重编译，graph 可能被打断，capture 也可能没法稳定复用。这样 timeline 上就会重新长出新的缝：有的是 graph break 造成的，有的是重新编译造成的，有的是某些原本能合并的路径因为 shape 抖动而重新被拆开。也就是说，shape 抖动不只是“多了一点编译时间”，而是会让本来已经收紧的 runtime 重新变得松散。

### 练习 2：compile cache toy

这道题最重要的，不是记住 exact key 和 bucketed key 这两个名字，而是理解：**为什么一点点 shape 抖动，最后会变成 timeline 上一阵一阵的慢。** 参考答案里，exact-key 模式一共出现了 7 次 compile，而 bucketed-key 模式只有 4 次。差别就出在：前者把每个不同 shape 都看成不同的 cache key，后者则允许一些相近的 sequence length 共用同一个 key。

**1）为什么 sequence length 的小幅抖动，也可能触发 exact-key 模式下的新编译？**

因为在 exact-key 的心智模型里，cache key 就是“这个 shape 本身”。于是 `(8, 1088)` 和 `(8, 1152)` 虽然只差了一点点，但只要 shape 不完全一样，它们就是两个不同 key。对编译器来说，这就意味着：之前编好的那份版本不一定能直接复用，于是要么 miss cache，要么重新生成新的特化版本。也正因为这样，真实 workload 里看似无害的 sequence length 抖动，常常会在 timeline 上表现成一阵阵编译热身和重新编译。

**2）为什么 bucketed / 更动态的策略能减少重编译，但也可能牺牲一部分优化空间？**

因为它的核心思路是：别把每个 shape 都看成完全不同，允许一批相近 shape 共用一个较粗的“代表 key”。这样做的直接好处是 cache hit 会更多，编译次数会更少；但代价是，生成出来的代码不再只服务于某一个精确 shape，而要适配一个更宽的范围。假设越宽，优化器能利用的静态信息就越少，某些专门化的 tile、访存布局或更激进的融合空间也可能变小。换句话说，它是在用一点“锐利度”去换更好的复用率。

**3）这个 bucketed toy 和真实 `torch.compile` dynamic shapes 有什么差别？**

差别很大。这个 toy 只是用“把 sequence length 向上取整到 bucket”来帮助读者建立直觉，而真实 `torch.compile` 的 dynamic shapes 不是简单做一个 round-up。真实系统通常还要考虑 guards、符号化 shape、哪些维度可以动态、哪些维度必须静态、graph break 会不会打断图、不同算子对动态 shape 的容忍度是否一致，以及后端到底能对这类动态范围做多少优化。也就是说，真实系统里的“能不能复用一个编译结果”远比这个 toy 更细，也更依赖上下文。

但这个 toy 仍然有一个很好的教学价值：它足够简单，能让我们先看清一件事——**shape 抖动并不会神秘地消失，它通常会先以 cache miss、重编译和 runtime 波动的形式出现在 timeline 上。** 一旦把这一层看懂，再去理解真实 `torch.compile` 的 dynamic shapes，就会容易很多。

## 本章 API 速记

- `torch.compile(module)`：尝试把模块里的可追踪计算收进图里，再交给编译后端做融合和代码生成。
- `torch.cuda.synchronize()`：等待当前 CUDA 工作完成；在 compile warm-up、计时和 profiler 场景里尤其常用。
- `torch.profiler.profile(...)`：采集一段运行过程中的 CPU / CUDA 事件，用来观察 timeline 是否连续、是否出现新的缝隙。
- `record_shapes=True`：把 shape 也记录下来，便于对照 shape 抖动是否带来 graph break、重编译或新的 runtime 波动。
- `prof.key_averages()`：把采集到的事件按名字聚合，方便看 compile 前后哪类事件变多、变少。
- `table(sort_by="self_cpu_time_total")`：按 CPU 自身时间排序，适合先看 compile、host 提交和 Python 侧开销。
- `prof.export_chrome_trace(...)`：导出时间线文件，之后可以在可视化工具里直接看 kernel 之间有没有空白、compile 之后有没有更连续。
- `torch.profiler.ProfilerActivity.CPU / CUDA`：分别控制是否记录 host 端和 device 端事件；两边一起看，才能判断问题到底在喂活还是在执行。

## 本章引用与延伸阅读
- [torch.compile — PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [torch.compiler — PyTorch documentation](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [Working with Graph Breaks — PyTorch documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
- [torch.cuda.graph — PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html)
- [torch.cuda.CUDAGraph — PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)
- [torch.profiler — PyTorch documentation](https://docs.pytorch.org/docs/stable/profiler.html)
- [Profiling to understand torch.compile performance — PyTorch documentation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)
- [flash-linear-attention — GitHub](https://github.com/fla-org/flash-linear-attention)
- [flame — GitHub](https://github.com/fla-org/flame)
