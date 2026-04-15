# 第五章：内存层级、带宽、吞吐与延迟

## Goal
- 建立这一部的成本模型：很多时候时间不是花在算，而是花在搬。
- 让读者学会先问“数据怎么移动”，再问“算子怎么算”。

## Reader Baseline
- 读者知道什么是矩阵乘法、归一化、embedding lookup 等常见操作。
- 不要求已经熟悉 roofline 或硬件性能分析术语。

## Core Questions
- memory hierarchy 为什么首先是一张成本图，而不是结构图？
- 带宽、吞吐、延迟、并行度分别在解释什么？
- arithmetic intensity / ops:byte 为什么能快速判断瓶颈方向？
- 为什么很多看起来简单的层反而最难提速？

## Section Arc
1. 开场问题：为什么简单 op 常常比大矩阵乘更像“系统问题”。
2. 把 memory hierarchy 改写成数据移动成本地图：谁快、谁慢、谁稀缺。
3. 带宽、吞吐、延迟：它们不是同义词，而是三种不同限制。
4. arithmetic intensity：怎样用“算多少、搬多少”判断 op 的性格。
5. matmul、softmax、layernorm、embedding lookup 的对照阅读。
6. profiler 视角下的 memory-bound 迹象：时间长、算不满、数据在搬。
7. 结尾收束：看到慢 op 时，先问它是算太少还是搬太多。

## Claims Requiring Current Verification
- 如果正文要写具体 GPU 的 HBM 带宽、L2 容量、shared memory 大小等数字，需要按当时官方规格核验。
- 如果正文要引用某一代 GPU 的 roofline 或 benchmark 图，需要按当时官方材料核验。

## Examples / Figures / Comparisons
- 贯穿例子：matmul、softmax、layernorm、embedding lookup。
- 优先用文字对比与 trace 观察，不主动画 roofline。
- 如确实需要图，可搜：`NVIDIA memory hierarchy gpu official`、`roofline model gpu`。

## Writing Guardrails
- 少用公式，多用直觉。
- roofline 只点到为止，不让本章变成性能建模教材。
- 首次出现就定义：latency、throughput、bandwidth、arithmetic intensity。
- 每一节都要落回“这会怎样出现在 trace 里”。

## Open Questions for User
- Chap5 要不要提前轻量引入 roofline，还是只保留 arithmetic intensity 直觉？
