这是一本写给 AI Researcher 的 AI Infra 学习笔记。

默认读者已经懂模型、优化和 RL，也会读论文、跑实验；更需要的是另一种观察角度：把“大模型系统”当成一个独立对象来研究。

所以这份笔记不按框架名来排目录，按系统瓶颈来排。主线很简单：
- 训练看显存、通信、编译和 checkpoint；
- 推理看 KV cache、prefill / decode、batching 和路由；
- RL 后训练看 rollout 数据流、资源编排和恢复；
- 最后回到 linear attention，讨论它在系统里的真实收益。

说明：当前大纲和许多初稿由 ChatGPT Pro、GPT-5.4 和 Codex 辅助生成。我会边学边改，所以把它当成工作草稿，不当成定稿。

# AI Infra：从分布式训练、推理系统到 Linear Attention

## 序：这本书到底在研究什么

第 1 章：AI Infra 的研究对象  
先把系统里真正要被放置、搬运、同步和回收的东西看清楚：参数、激活、优化器状态、KV cache、网络流量、调度和时间。

第 2 章：什么叫系统视角  
把一次训练或一次生成放回时间里，看数据流、内存流、通信流和控制流怎样交错。

第 3 章：本书的方法论  
先判断瓶颈，再看框架；先做 profiling，再动手优化；先看 wall-clock，再谈复杂度。

这几章的作用，是把“会看模型”慢慢变成“会看系统”。

---

## 第一部：硬件、运行时与 profiling

这一部是地基。先把 GPU 怎么干活看清楚：内存层级是什么，带宽和延迟各卡在哪里，kernel 怎么发起，stream 怎样交错，collective 为什么会把多卡训练变成通信问题。把这些东西看明白，后面的优化才不会漂在空中。

第 4 章：GPU 不是黑盒  
第 5 章：内存层级、带宽、吞吐与延迟  
第 6 章：kernel、stream 与 launch overhead  
第 7 章：collective 的语义：all-reduce、all-gather、reduce-scatter、all-to-all  
第 8 章：profiling 的方法：timeline、kernel metric、热点定位  
第 9 章：为什么很多模型问题最后都会变成 IO 问题

读完这一部，你至少应该能看懂一张 profiler trace，知道时间花在计算、通信、调度还是数据输入上。

---

## 第二部：训练 Infra 的最小闭环

这一部用 PyTorch 训练栈做主线。到 2026，通用训练底座已经很像一套稳定组合：`torch.distributed`、FSDP2、DCP、activation checkpointing、`torch.compile`，再加上 Megatron Core 提供的 TP / PP / CP / EP 等并行手段。这里更关心训练状态怎样被切分、重组、同步和保存，而不是去背 API。

第 10 章：`torch.distributed` 与 process group  
第 11 章：DDP 作为一切分布式训练的参照物  
第 12 章：FSDP2：参数、梯度、optimizer state 的时空重组  
第 13 章：Distributed Checkpoint：重启、reshard、容错  
第 14 章：activation checkpointing 与显存-算力交换  
第 15 章：`torch.compile`：图捕获、specialization 与 graph break  
第 16 章：dynamic shapes：为什么“可变长度”常常破坏理想加速  
第 17 章：Megatron Core：TP、PP、CP、EP 的统一视图  
第 18 章：MoE 的系统问题：routing、负载不均、专家并行

这部分读完，你应该能回答一个很实际的问题：面对一个训练任务，为什么该用 DDP，什么时候该上 FSDP2，什么时候又得把 TP / PP / CP / EP 一起拉进来。

---

## 第三部：Attention 作为系统对象

这一部单独看 attention，不过切入点放在实现和系统代价上。FlashAttention 让人看清 exact attention 还有多少 IO 空间可挖；FlexAttention 则像一座桥，把研究里的 attention 变体接进 PyTorch 编译栈。学这一部，目的是把“attention 很贵”这句话拆开，拆到你能说清楚它究竟贵在访存、kernel、编译，还是并行方式。

第 19 章：从 vanilla SDPA 到 FlashAttention  
第 20 章：为什么 FlashAttention 的关键词是 IO-aware，不是 approximation  
第 21 章：FlashAttention-2/3 的并行化与硬件感知  
第 22 章：FlexAttention：把 mask、bias、window、文档边界写成可编译接口  
第 23 章：Attention Gym：把新注意力做成可试验对象  
第 24 章：长上下文训练中的 packed sequence、block mask 与 context parallel  
第 25 章：什么时候 exact attention 已经足够好，不必急着 linearize

这一部读完，你不只会说 attention 成本高，还能说清成本落在哪一层，以及该从 kernel、compile 还是并行策略下手。

---

## 第四部：推理 Infra 与 KV Cache 世界

这一部是全书重心之一。到 2026，推理系统的中心对象已经很明确：KV cache。模型 forward 当然还在，但真正决定系统形态的，往往是 KV cache 的建立、复用、迁移和回收。prefill / decode 的分工，continuous batching 的调度，prefix caching 的收益，乃至跨实例复用和 KV-aware routing，本质上都在围着这件事转。

第 26 章：KV cache 的数学语义与生命周期  
第 27 章：Dynamic Cache、Static Cache、offloading 与 compile 的关系  
第 28 章：vLLM：PagedAttention 与 continuous batching  
第 29 章：prefix caching：为什么共享前缀常常能直接省下一大段成本  
第 30 章：SGLang：RadixAttention 与 shared-prefix runtime  
第 31 章：prefill 与 decode 为什么要分开看  
第 32 章：disaggregated prefill / decode 的体系结构  
第 33 章：LMCache：prefill once, reuse everywhere  
第 34 章：Dynamo：KV-aware routing、多层缓存与数据中心级编排  
第 35 章：serving 评测：TTFT、TPOT、吞吐、并发、SLO

这一部读完，你应该能把“长上下文推理慢”拆成几件具体的事：prefill 计算、decode 带宽、KV 占用、prefix reuse、batching 策略和路由策略。

---

## 第五部：RL 不再当算法学，而当系统学

你已经懂 PPO、DPO、GRPO 这类算法，所以这一部不重讲公式，重点放在系统闭环：谁负责生成，谁负责打分，谁负责训练，这些组件怎样放到 GPU 上，怎样减少等待和空转。TRL 适合当最小基线；OpenRLHF 展示的是 Ray + vLLM + DeepSpeed 这类系统拼装；verl 与 HybridFlow 则把 RLHF / RLVR 写成显式数据流。

第 36 章：把 RL post-training 画成数据流图  
第 37 章：actor、reference、reward、critic、sampler 的拓扑  
第 38 章：在线生成为什么常常是瓶颈  
第 39 章：TRL：作为最小可运行基线的价值  
第 40 章：OpenRLHF：Ray、vLLM、DeepSpeed 的系统拼装  
第 41 章：Hybrid Engine：如何让训练和 rollout 共享 GPU 而不闲置  
第 42 章：verl 与 HybridFlow：把 RL 系统写成可重组 dataflow  
第 43 章：异步模式、队列、背压与吞吐  
第 44 章：RL 系统的 profiling：idle time、reshard、通信与生成抖动  
第 45 章：为什么 RL Infra 的核心问题是 orchestration，而不是 loss function

这一部读完，你应该能独立检查一个 RLHF / RLVR 框架：瓶颈究竟在 rollout、打分、训练、通信，还是资源编排。

---

## 第六部：Linear Attention 主线

这部分会放在后半本。原因很简单：先把 exact attention 在系统层面已经被优化到什么程度看清楚，才有资格判断 linear attention 的真实价值。纯 kernelized linear attention 只是起点；后面的 fast-weight 视角、RetNet / Mamba 这类 recurrent / SSM 路线、以及 DeltaNet / Gated DeltaNet / KDA 等更强调状态更新表达力的方法，才把这条线真正拉开。与此同时，近年的证据越来越指向 hybrid linear attention，而不是“全模型纯 linear”。

第 46 章：为什么人们要 linearize attention  
第 47 章：从复杂度神话到系统现实  
第 48 章：Performer 与 kernelized linear attention 的起点  
第 49 章：Fast Weight 视角：为什么线性注意力像可编程记忆  
第 50 章：S4 与状态空间模型的基本语言  
第 51 章：RetNet：并行训练、递归推理、chunkwise recurrence  
第 52 章：Mamba：selective state spaces 与硬件感知实现  
第 53 章：DeltaNet：delta rule 与沿序列维并行训练  
第 54 章：Gated DeltaNet：为什么 gating 改变了表达力边界  
第 55 章：Hybrid linear attention：线性层和 full attention 层如何配比  
第 56 章：What Matters in Linearizing Language Models：哪些东西不能靠算力硬补  
第 57 章：Kimi Linear：KDA 与 hybrid 设计  
第 58 章：`flash-linear-attention`：把 GDN / KDA 变成可实验系统  
第 59 章：系统现实：decode 带宽、recurrent state 与“线性不等于更快”

这一部最重要的认知升级有三点。第一，线性注意力真正难的地方，在记忆能力和检索能力能否保住。第二，前沿路线更像 hybrid，而不是纯 linear。第三，就算模型层面很好看，decode 也可能仍然是 memory-bound；线性复杂度不会自动换来更快的真实系统。

---

## 第七部：前沿整合与研究方法

第 60 章：如何公平比较 full attention、hybrid、linear、SSM  
第 61 章：训练指标与部署指标为什么常常不一致  
第 62 章：从 paper claim 到 system claim：该验证什么  
第 63 章：做 ablation 的顺序  
第 64 章：从单卡原型到多卡实验  
第 65 章：如何写一篇 AI Infra 风格的研究报告
