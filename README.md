该Repo是面向AI Researcher学习AI Infra的个人学习笔记。
这本书的定位应该很明确：**你已经懂模型、优化、RL 算法，但还没有把“大模型系统”当成一个独立对象来研究**。因此它不该按“框架名”组织，而该按**系统瓶颈**组织：显存与通信、kernel 与编译、KV cache 与 serving、RL rollout 数据流、以及最终的 subquadratic sequence models。到 2026，这条主线已经比较清晰：训练侧是 PyTorch 的 FSDP2、DCP、`torch.compile`、FlexAttention 与 Megatron Core 等多维并行工具；推理侧围绕 KV cache、prefix reuse、continuous batching、disaggregated serving 展开；RL 后训练侧越来越像一个分布式数据流系统，TRL 是轻量基线，OpenRLHF 和 verl/HybridFlow 更偏系统化编排。([PyTorch Documentation][1])

Caution: 大纲和文字内容纯由ChatGPT Pro mode和GPT 5.4 extra high thinking + Codex生成，will be examined as I keep learning.

# AI Infra：从分布式训练、推理系统到 Linear Attention

## 序：这本书到底在研究什么

第 1 章：AI Infra 的研究对象
不是“模型结构”本身，而是参数、激活、优化器状态、KV cache、网络流量、调度和时间。

第 2 章：什么叫系统视角
把一次训练或一次生成拆成数据流、内存流、通信流和控制流。

第 3 章：本书的方法论
先建立瓶颈模型，再学框架；先做 profiling，再谈优化；先看 wall-clock，再看理论复杂度。

这一序章的目标，是把你从“会看论文”切换到“会问系统问题”。

---

## 第一部：硬件、运行时与 profiling

这一部是全书地基。你要先建立 GPU memory hierarchy、kernel launch、occupancy、HBM/SRAM、stream、collective communication 的直觉，否则后面所有优化都会变成“背术语”。NCCL 的 collectives、PyTorch 的编译与动态 shape、以及 FlashAttention 的 IO-aware 视角，构成这部分的技术骨架。([NVIDIA Docs][3])

第 4 章：GPU 不是黑盒
第 5 章：内存层级、带宽、吞吐与延迟
第 6 章：kernel、stream 与 launch overhead
第 7 章：collective 的语义：all-reduce、all-gather、reduce-scatter、all-to-all
第 8 章：profiling 的方法：timeline、kernel metric、热点定位
第 9 章：为什么很多模型问题最后都会变成 IO 问题

这一部读完，你应该能看懂一张 profiler trace，并说出瓶颈是在算子、调度、通信还是数据输入。

---

## 第二部：训练 Infra 的最小闭环

这一部以 PyTorch 为主线，因为 2026 的通用训练基座，已经很大程度上收敛到 `torch.distributed`、FSDP2、DCP、activation checkpointing 与 `torch.compile` 的组合；而更大规模、更复杂的并行，则交给 Megatron Core 的 TP/PP/CP/EP 体系。([PyTorch Documentation][1])

第 10 章：`torch.distributed` 与 process group
第 11 章：DDP 作为一切分布式训练的参照物
第 12 章：FSDP2：参数、梯度、optimizer state 的时空重组
第 13 章：Distributed Checkpoint：重启、reshard、容错
第 14 章：activation checkpointing 与显存-算力交换
第 15 章：`torch.compile`：图捕获、specialization 与 graph break
第 16 章：dynamic shapes：为什么“可变长度”常常破坏理想加速
第 17 章：Megatron Core：TP、PP、CP、EP 的统一视图
第 18 章：MoE 的系统问题：routing、负载不均、专家并行

这部分的核心不是“会用 API”，而是你能解释：为什么某个训练任务该用 DDP、FSDP2，还是 TP/PP/CP/EP 的组合。

---

## 第三部：Attention 作为系统对象

这一部专门研究 attention，不是从公式出发，而是从实现出发。FlashAttention 系列证明了 exact attention 仍然有大量系统优化空间；FlexAttention 则把自定义 attention 变体放进了 PyTorch 编译栈，成为 2026 非常值得掌握的“研究-工程桥梁”。([arXiv][2])

第 19 章：从 vanilla SDPA 到 FlashAttention
第 20 章：为什么 FlashAttention 的关键词是 IO-aware，不是 approximation
第 21 章：FlashAttention-2/3 的并行化与硬件感知
第 22 章：FlexAttention：把 mask、bias、window、文档边界写成可编译接口
第 23 章：Attention Gym：把新注意力做成可试验对象
第 24 章：长上下文训练中的 packed sequence、block mask 与 context parallel
第 25 章：什么时候 exact attention 已经足够好，不必急着 linearize

这部分读完，你不只是“知道 attention 很贵”，而是能判断它为什么贵、贵在哪一层、以及应该去改 kernel、compile 还是并行策略。

---

## 第四部：推理 Infra 与 KV Cache 世界

这一部是全书的重心之一。到 2026，推理系统的真正主角已经不是“模型 forward”本身，而是 **KV cache 的生命周期管理**。Hugging Face 已经把 cache 抽象分成不同策略，并明确区分哪些 cache 适合 `torch.compile`；vLLM 的重点是 PagedAttention、continuous batching、chunked/disaggregated prefill 和 prefix caching；SGLang 的重点是 RadixAttention 与 PD disaggregation；LMCache 与 Dynamo 则把 KV reuse、分层缓存、路由与跨实例复用进一步系统化。([Hugging Face][4])

第 26 章：KV cache 的数学语义与生命周期
第 27 章：Dynamic Cache、Static Cache、offloading 与 compile 的关系
第 28 章：vLLM：PagedAttention 与 continuous batching
第 29 章：prefix caching：为什么共享前缀是“几乎免费的午餐”
第 30 章：SGLang：RadixAttention 与 shared-prefix runtime
第 31 章：prefill 与 decode 为什么应该分开看
第 32 章：disaggregated prefill / decode 的体系结构
第 33 章：LMCache：prefill once, reuse everywhere
第 34 章：Dynamo：KV-aware routing、多层缓存与数据中心级编排
第 35 章：serving 评测：TTFT、TPOT、吞吐、并发、SLO

这一部读完，你应该能把“长上下文推理慢”具体拆成：prefill 计算、decode 带宽、KV 占用、prefix reuse、batching 策略和路由策略。

---

## 第五部：RL 不再当算法学，而当系统学

既然你对 RL 算法和对象都熟，这一部就不重复 PPO、DPO、GRPO 的数学，而把重点放在：**谁在生成、谁在打分、谁在训练、这些组件怎么放到 GPU 上，以及资源怎么不空转**。TRL 提供单机/单集群的最小基线；OpenRLHF 的特征是 Ray + vLLM + DeepSpeed 的系统组合，以及 hybrid engine 以减少资源空闲；verl 的重点则是 HybridFlow，把 RLHF/RLVR 看成显式数据流图来编程。([Hugging Face][5])

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

这一部读完，你应该能独立审视一个 RLHF/RLVR 框架：它的瓶颈到底在 rollout、打分、训练、通信，还是资源编排。

---

## 第六部：Linear Attention 主线

这部分才是你最关心的重点，但我会把它放在后半本。理由很简单：你需要先知道 exact attention 在系统上已经被优化到了什么程度，才能判断 linear attention 的真实价值。当前研究已经很清楚：纯 kernelized linear attention 只是起点；更强的路线包括 fast-weight 视角、RetNet/Mamba 这类 recurrent/SSM 视角、以及 DeltaNet/Gated DeltaNet/KDA 这类更强调状态更新表达力的路线。与此同时，2025–2026 的证据也越来越倾向于 **hybrid linear attention**，而不是“全模型纯 linear”。([arXiv][6])

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
第 58 章：`flash-linear-attention`：把 GDN/KDA 变成可实验系统
第 59 章：系统现实：decode 带宽、recurrent state 与“线性不等于更快”

这一部最重要的认知升级有三个。第一，线性注意力真正难的不是把复杂度写成 (O(n))，而是**如何保住记忆能力与检索能力**。第二，当前更可信的前沿是 hybrid，而不是纯 linear。系统性比较工作给出的经验区间，倾向于 HGRN-2 或 GatedDeltaNet 这类更强线性层，并在大约 3:1 到 6:1 的线性层/全注意力层配比内寻找平衡。第三，即使模型层面很漂亮，decode 也可能仍然是 memory-bound；最近针对 Gated DeltaNet 的系统论文就直接指出，batch-1 GPU decode 仍可能被 recurrent state 的 HBM 往返压住。([arXiv][7])

---

## 第七部：前沿整合与研究方法

第 60 章：如何公平比较 full attention、hybrid、linear、SSM
第 61 章：训练指标与部署指标为什么常常不一致
第 62 章：从 paper claim 到 system claim：该验证什么
第 63 章：做 ablation 的顺序
第 64 章：从单卡原型到多卡实验
第 65 章：如何写一篇 AI Infra 风格的研究报告
