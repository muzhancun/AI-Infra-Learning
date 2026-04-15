# 第六章：kernel、stream 与 launch overhead

## Goal
- 建立 runtime 直觉：很多性能问题不在 kernel 本身，而在 host 如何把 GPU 喂饱。
- 让读者第一次把 eager、compile、dynamic shape、graph break 看成 timeline 形状的变化。

## Reader Baseline
- 读者已经接受“模型层会落成 kernel”的基本图景。
- 不要求提前熟悉 PyTorch 编译栈内部实现。

## Core Questions
- 什么叫 launch，为什么 launch 本身会变成成本？
- stream 是什么，不是什么？
- 小 kernel 为什么会把时间浪费在 timeline 的缝隙里？
- `torch.compile`、dynamic shapes、graph breaks 会怎样改变 runtime？
- 什么时候值得引入 CUDA Graphs？

## Section Arc
1. 开场问题：为什么 GPU 明明不慢，程序还是跑得断断续续。
2. 什么是 launch：从 Python/host 到 device 执行的那段路。
3. stream 的语义：顺序队列、依赖关系与可重叠的边界。
4. launch overhead 与小 kernel：为什么“碎”会浪费时间。
5. `torch.compile` 的 runtime 后果：fusion、graph break、recompilation、shape 抖动。
6. CUDA Graphs 解决什么问题，不解决什么问题。
7. 用 eager / compile / 固定 shape / 动态 shape 四组 trace 做一次诊断练习。
8. 结尾收束：timeline 的缝隙本身就是重要信号。

## Claims Requiring Current Verification
- `torch.compile`、dynamic shapes、graph breaks、CUDA Graphs 在写作时的官方表述与限制需要按当时 PyTorch 文档核验。
- 如果正文要写当前推荐的 compile / CUDAGraph 工作流，需要按当时版本核验。

## Examples / Figures / Comparisons
- 贯穿例子：eager vs `torch.compile`；固定 shape vs 动态 shape。
- 优先用 timeline 描述，不主动配图。
- 如确实需要图，可搜：`PyTorch compile graph breaks`、`CUDA Graphs official timeline`。

## Writing Guardrails
- 这一章讲 runtime 后果，不讲完整编译器架构史。
- 少讲 API，多讲 timeline 发生了什么。
- 首次出现就定义：launch overhead、stream、graph break、recompilation、CUDA Graphs。
- 避免与第二部更细的 `torch.compile` 章节重复。

## Open Questions for User
- 这一章里 `torch.compile` 你想停在 runtime 后果，还是要顺带放一小段 TorchDynamo / AOTAutograd / Inductor 名字？
