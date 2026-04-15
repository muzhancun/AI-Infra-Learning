# 第一部：硬件、运行时与 profiling

## Goal
- 把这一部写成“从 trace 反推系统”的入门，而不是 CUDA 名词表。
- 给读者建立一套统一的瓶颈语言：算子、内存、调度、通信、数据流，最后都能落回 profiler。
- 读完后，读者至少能把“GPU 利用率不高”拆成更具体的几类原因：算力受限、访存受限、launch/host 受限、通信等待、数据输入没跟上。

## Reader Baseline
- 读者已经懂模型、优化、RL，也熟悉训练 / 推理的基本流程。
- 不需要从零重讲 Transformer、矩阵乘法、反向传播或优化器公式。
- 这一部要补的是：GPU 机器直觉、runtime 直觉、profiling 方法，以及“系统瓶颈如何表现在 trace 上”。

## Core Questions
- 一个 op 为什么会快，为什么会慢？
- 为什么“GPU utilization 低”常常不是结论，而只是待分解的症状？
- 从单卡 kernel 到多卡 collective，如何用统一的成本 / 执行模型解释性能？
- 怎样把一张 trace 变成可验证的性能假设？
- 为什么很多看起来像“模型层”的问题，最后会落到 IO 与数据移动上？

## Recommended File Structure

```text
硬件、运行时与 profiling/
  AGENTS.md
  Chap4/
    AGENTS.md
    第四章：GPU 不是黑盒.md
  Chap5/
    AGENTS.md
    第五章：内存层级、带宽、吞吐与延迟.md
  Chap6/
    AGENTS.md
    第六章：kernel、stream 与 launch overhead.md
  Chap7/
    AGENTS.md
    第七章：collective 的语义：all-reduce、all-gather、reduce-scatter、all-to-all.md
  Chap8/
    AGENTS.md
    第八章：profiling 的方法：timeline、kernel metric、热点定位.md
  Chap9/
    AGENTS.md
    第九章：为什么很多模型问题最后都会变成 IO 问题.md
```

说明：先维持和 `[[AI Infra/序章]]` 一致的 `ChapX/AGENTS.md + 正文章节文件` 结构；如果后面要统一放图和 trace，再补一个共享附件目录。

## Section Arc
1. 先建立机器模型：GPU 到底在执行什么。
2. 再建立成本模型：时间为什么常常花在“搬”而不是“算”。
3. 然后进入执行模型：kernel / stream / launch 怎样把算子变成 timeline。
4. 再进入通信模型：collective 在多卡里到底在交换什么。
5. 然后给观测模型：怎样用 profiler 把症状变成定位。
6. 最后用 IO 视角收束，让前五章变成一套统一世界观。

## Shared Chapter Pattern
- 每章尽量固定四个动作：反直觉问题 → 最小实验 → 读一张真实 trace → 收束成一个系统结论。
- 每章都回答“这章新增了哪一种瓶颈语言”。
- 每章只做一个核心认知升级，不做 API 罗列。
- 可以有“2026 前沿侧栏”，但必须和正文主线分开，且写作时再做现行版本核验。

## Chapter Outlines

### Chap4 / 第四章：GPU 不是黑盒
**Chapter Promise**
- 让读者第一次形成 GPU 的最小机器直觉：GPU 看见的不是“层”，而是 kernel、线程、内存访问和资源约束。

**Core Questions**
- PyTorch 里的一个 op 落到 GPU 上，实际变成了什么？
- SM、warp、block 分别在扮演什么角色？
- register、shared memory、block size 为什么会一起限制并发？
- occupancy 为什么只是必要条件，而不是性能结论？

**Section Arc**
1. 从 PyTorch op 到 kernel：GPU 不执行“模型层”，只执行 kernel 队列。
2. SM、warp、block 的最小心智模型：先讲谁在执行，再讲谁在被调度。
3. global memory 与 on-chip memory：把结构图改写成“谁贵、谁快、谁稀缺”。
4. 资源约束如何限制并发：register、shared memory、block size、active warps。
5. 为什么 occupancy 不是万能解释：高 occupancy 不自动等于高性能。
6. 用 matmul、layernorm、relu 建立第一轮“算力受限 / 访存受限 / kernel 太碎”直觉。

**Examples / Figures / Comparisons**
- 图：SM / warp / block 关系图。
- 图：一次 op 到多个 kernel 的映射示意。
- 对比：matmul vs layernorm vs relu。

**Writing Guardrails**
- 不陷进过深的 CUDA 语法细节。
- 不按 GPU 部件清单讲，而按“为什么会快 / 慢”讲。

### Chap5 / 第五章：内存层级、带宽、吞吐与延迟
**Chapter Promise**
- 建立这一部的成本模型：很多时间不是花在算，而是花在搬。

**Core Questions**
- memory hierarchy 为什么首先是成本图，而不是结构图？
- 带宽、吞吐、延迟、并行度分别回答什么问题？
- arithmetic intensity / ops:byte 为什么能快速判断瓶颈方向？
- 为什么简单层常常比大矩阵乘更难提速？

**Section Arc**
1. 从“内存层级示意图”切到“数据移动成本地图”。
2. 带宽、吞吐、延迟：三者如何共同决定体感速度。
3. arithmetic intensity：怎样用“算多少、搬多少”判断 op 的性格。
4. 为什么 elementwise、reduction、normalization 常常 memory-bound。
5. 用 matmul、softmax、layernorm、embedding lookup 做定性判断。
6. 收束：看到慢 op 时，先问“算太少”还是“搬太多”。

**Examples / Figures / Comparisons**
- 图：register / shared / L2 / HBM 的成本金字塔。
- 对比：compute-bound vs memory-bound op。
- 实验：同一批 op 在 trace 里的形态差异。

**Writing Guardrails**
- 少上公式，先让读者形成成本直觉。
- roofline 可以点到为止，不要让这一章变成数学证明。

### Chap6 / 第六章：kernel、stream 与 launch overhead
**Chapter Promise**
- 让读者理解：很多“GPU 没跑满”不是 kernel 本身慢，而是 runtime 没把 GPU 连续喂饱。

**Core Questions**
- 什么叫 launch，它为什么会成为成本？
- stream 是什么，不是什么？
- 小 kernel 为什么会把时间浪费在 timeline 的缝隙里？
- eager、`torch.compile`、dynamic shapes、graph breaks 分别怎样改写 timeline？
- 什么时候该把 CUDA Graphs 引进来？

**Section Arc**
1. 什么是 launch：从 host 发起，到 device 开始执行。
2. stream 是顺序队列，不是“自动并行按钮”。
3. launch overhead 与小 kernel：为什么碎片化执行会浪费时间。
4. `torch.compile` 的 runtime 后果：fusion、graph breaks、recompilation、shape 抖动。
5. CUDA Graphs 解决什么问题，不解决什么问题。
6. 用 eager / compile / 固定 shape / 变化 shape 四种 trace 做第一次 runtime 诊断。

**Examples / Figures / Comparisons**
- 图：有缝隙的 timeline vs 被喂满的 timeline。
- 对比：eager vs `torch.compile`。
- 对比：固定 shape vs 动态 shape。

**Writing Guardrails**
- 这一章只讲 compile 的运行时后果，不展开到后面第二部会系统讲的编译栈细节。
- 核心不是 API，而是 timeline 怎么变。

### Chap7 / 第七章：collective 的语义：all-reduce、all-gather、reduce-scatter、all-to-all
**Chapter Promise**
- 把多卡通信先讲成“张量所有权和语义变化”，再讲成“性能问题”。

**Core Questions**
- 四个最重要的 collective，分别是谁给谁、谁拿到完整结果、谁只保留分片？
- 参数同步、激活重分布、MoE token 路由分别对应哪些 collective？
- 为什么很多并行策略本质上只是 collective 组合？
- 什么叫“语义正确但性能错误”的通信？

**Section Arc**
1. 先画 4 卡例子，不先讲 ring / tree。
2. all-reduce：每个人都拿到归约后的完整结果。
3. all-gather / reduce-scatter：完整张量与分片张量的来回变形。
4. all-to-all：数据拥有者与消费者同时改变。
5. 把 DDP、TP、CP、EP、MoE 路由映射回 collective 语义。
6. 收束：通信问题首先是语义问题，其次才是实现问题。

**Examples / Figures / Comparisons**
- 图：四种 collective 的 4 卡数据流示意。
- 对比：all-reduce vs reduce-scatter + all-gather。
- 例子：参数同步、序列切分、专家路由。

**Writing Guardrails**
- 本章先稳住语义，不急着扩展到 NCCL 算法细节。
- 只在确实需要时才引出 ring / tree / topology。

### Chap8 / 第八章：profiling 的方法：timeline、kernel metric、热点定位
**Chapter Promise**
- 给读者一套“先看哪里、再下钻哪里”的 profiling 工作流，而不是工具菜单。

**Core Questions**
- 为什么应该先看 timeline，再看 kernel metric？
- host 空转、device 空转、通信等待、data input 等待怎样区分？
- PyTorch Profiler、Nsight Systems、Nsight Compute 各自回答什么问题？
- 单机 trace 和多卡 trace 的读法差别在哪里？

**Section Arc**
1. profiling 的第一原则：先回答“时间花在哪里”。
2. timeline 的基本读法：连续、空洞、重叠、阻塞、等待。
3. 什么时候留在 PyTorch Profiler，什么时候上 Nsight Systems。
4. 什么时候必须下钻到 Nsight Compute 看 kernel metric。
5. 单机 trace、多卡 trace、data loader 卡住的 trace 各怎么看。
6. 收束成一个闭环：假设 → profile → 改动 → 再 profile。

**Examples / Figures / Comparisons**
- 图：单机 trace 标注图。
- 图：多卡 trace 标注图。
- 对比：PyTorch Profiler vs Nsight Systems vs Nsight Compute。
- 练习：给三张 trace，各用一句话诊断。

**Writing Guardrails**
- 不把工具介绍写成参数手册。
- 优先教“如何判断”，再教“点哪个按钮”。

### Chap9 / 第九章：为什么很多模型问题最后都会变成 IO 问题
**Chapter Promise**
- 用 IO 视角把前五章真正串起来，让读者看到“模型、kernel、内存、调度、通信”其实都在争夺数据移动预算。

**Core Questions**
- 为什么 attention 不只是计算复杂度问题，也是 memory traffic 问题？
- 为什么很多 elementwise / reduction / normalization 问题天然带 IO 气质？
- FlashAttention 为什么适合作为这一部的收束例子？
- prefill 与 decode 为什么是两种不同的 IO 问题？

**Section Arc**
1. 回看前几章：memory-bound、launch-bound、comm-bound，本质都与数据流有关。
2. 从 layernorm、softmax、embedding 看“简单层”的 IO 性格。
3. attention 为什么既是 O(n^2) 问题，也是 HBM 读写问题。
4. 用 FlashAttention 作为“算法 × memory hierarchy × kernel scheduling”的示范。
5. 提前埋伏笔：prefill、decode、KV cache 会把 IO 问题进一步放大。
6. 结尾收束：真正成熟的系统直觉，是先问数据在什么时候、从哪里、搬到哪里。

**Examples / Figures / Comparisons**
- 图：标准 attention vs IO-aware attention 的读写路径。
- 对比：prefill vs decode 的资源画像。
- 对比：算力瓶颈叙事 vs IO 瓶颈叙事。

**Writing Guardrails**
- 这一章是世界观收束，不是 FlashAttention 实现细节大全。
- 重点是统一视角，为第三部和第四部埋线。

## Claims Requiring Current Verification
- `torch.compile`、dynamic shapes、graph breaks、CUDA Graphs 在当前 PyTorch 版本里的表述与限制。
  - Why time-sensitive: PyTorch 编译栈和文档表述变化快。
  - Best source type: PyTorch 官方文档 / release notes / 官方博客。
- PyTorch Profiler、Perfetto、TensorBoard trace viewer、HTA 的推荐工作流。
  - Why time-sensitive: 官方推荐工具链和入口会变。
  - Best source type: PyTorch 官方 profiler 文档 / tutorial。
- differentiable collectives、FlexAttention backend、FlashAttention-4 等 2026 侧栏内容。
  - Why time-sensitive: 都属于版本与路线图敏感内容。
  - Best source type: PyTorch 官方 release blog / 官方文档 / 论文或仓库说明。
- Hopper / Blackwell 等架构特性若被用作“当前主流实践”举例。
  - Why time-sensitive: 架构代际与最佳实践变化快。
  - Best source type: NVIDIA 官方文档 / 官方技术博客。

## Examples / Figures / Comparisons
- 贯穿示例建议：`matmul / layernorm / softmax / all-reduce / dataloader gap` 五件套，跨章节复用。
- 贯穿图示建议：
  - GPU 机器模型图
  - 内存层级成本图
  - runtime timeline 图
  - collective 数据所有权图
  - profiling 决策树
  - IO 统一世界观图
- 贯穿比较建议：
  - compute-bound vs memory-bound
  - eager vs compile
  - fixed shape vs dynamic shape
  - all-reduce vs reduce-scatter + all-gather
  - PyTorch Profiler vs Nsight Systems vs Nsight Compute

## Writing Guardrails
- 默认 voice：像黑板前的耐心专家，不像 API 索引。
- 先给具体图景，再上抽象术语。
- unfamiliar terms 要贴着第一次出现的位置定义：occupancy、arithmetic intensity、graph break、collective、timeline gap、TTFT 等。
- 尽量少做大段 itemize；只有在“结构本身就是重点”时才列表。
- 每章都要把知识落回“如何看 trace、如何形成假设、如何做最小实验”。
- 避免和第二部的 `torch.compile` / dynamic shapes 章节重复；这一部只保留它们的 runtime 与 profiling 视角。

## Open Questions for User
- Chap6 里 `torch.compile` 你想保留到什么深度：只讲 timeline 后果，还是也预埋一小段编译栈名字？
- Chap7 你更想停留在 collective 语义，还是顺手带一点 ring / tree / topology 的性能直觉？
- Chap8 你希望 profiling 以单机为主，还是提前把分布式 trace 和 HTA 的入口放进来？
- Chap9 结尾你更想收在 FlashAttention，还是提前更明显地引到 prefill / decode / KV cache？

## Current Editorial Decisions
- Chap6：以 runtime / timeline 为主线，轻量点名 TorchDynamo / AOTAutograd / TorchInductor，但不展开实现细节。
- Chap7：以 collective 语义为主，结尾补少量 ring / tree / topology 的性能直觉。
- Chap8：单机 profiling 作为主线，结尾自然引出分布式 trace 与 HTA。
- Chap9：用 FlashAttention 收束，并更明显地把读者带向 prefill / decode / KV cache。

## Practice Scripts
- Chap4: `Chap4/scripts/occupancy_toy.py`, `Chap4/scripts/block_scheduler_toy.py`
- Chap5: `Chap5/scripts/arithmetic_intensity_toy.py`, `Chap5/scripts/roofline_toy.py`
- Chap6: `Chap6/scripts/timeline_gaps.py`, `Chap6/scripts/compile_cache_toy.py`
- Chap7: `Chap7/scripts/collective_semantics.py`, `Chap7/scripts/collective_cost_toy.py`
- Chap8: `Chap8/scripts/trace_accounting.py`, `Chap8/scripts/rank_skew_toy.py`
- Chap9: `Chap9/scripts/attention_io_toy.py`, `Chap9/scripts/kv_cache_budget.py`
- 每个 starter 都配了一个 `_answer.py` 参考答案。
