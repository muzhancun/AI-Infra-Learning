# 第四章：GPU 不是黑盒

## Goal
- 让读者建立 GPU 的最小机器模型：GPU 执行的不是“层”，而是 kernel、线程、内存访问与资源约束。
- 把“一个 op 为什么快/慢”拆成几个可操作的问题：算力、访存、launch、资源占用、并发不足。

## Reader Baseline
- 读者懂模型、训练 loop、矩阵乘法和常见算子。
- 不要求读者已经写过 CUDA kernel。
- 本章不重讲深度学习基础，而是补“机器直觉”。

## Core Questions
- PyTorch 里的一个 op 落到 GPU 上，实际变成了什么？
- SM、warp、block 各自负责什么？
- register、shared memory、block size 为什么会一起限制并发？
- occupancy 为什么不是性能的直接同义词？

## Section Arc
1. 开场问题：为什么同样是“一个层”，在 profiler 里却会变成完全不同的 kernel 形状。
2. 从 op 到 kernel：GPU 不认识模型层，只认识被发射的 kernel。
3. SM、warp、block 的最小心智模型：谁执行、谁调度、谁必须彼此独立。
4. global memory、register、shared memory：不是结构清单，而是成本与稀缺性的差别。
5. occupancy、资源约束与并发：为什么线程够不够、资源满不满会改变结果。
6. 用 matmul、layernorm、relu 做第一次性能分类：算力受限、访存受限、kernel 太碎。
7. 结尾收束：GPU utilization 低不是结论，只是待分解的症状。

## Claims Requiring Current Verification
- 如果正文要写具体 GPU 代际（如 Hopper、Blackwell）的资源上限、调度差异或 cache 特性，需要按当时官方文档核验。
- 如果正文要举“当前 PyTorch 会把某个 op 融成哪些 kernel”的版本相关例子，需要按当时版本核验。

## Examples / Figures / Comparisons
- 贯穿例子：matmul、layernorm、relu。
- 优先用文字和 trace 解释，不主动配图。
- 如确实需要图，可搜：`CUDA SM warp block diagram`、`CUDA memory hierarchy official`。

## Writing Guardrails
- 不把本章写成 CUDA 术语表。
- 不陷入过深的架构细节或 PTX 细节。
- 每个概念都要回到“为什么会影响速度”。
- 首次出现就定义：SM、warp、block、occupancy。

## Open Questions for User
- 是否要在本章末尾放一个很小的“读 kernel 名字”练习，还是把练习都留到 profiling 章？
