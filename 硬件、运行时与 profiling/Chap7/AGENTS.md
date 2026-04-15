# 第七章：collective 的语义：all-reduce、all-gather、reduce-scatter、all-to-all

## Goal
- 先把 collective 讲成“张量所有权如何变化”，再讲成通信性能问题。
- 让读者能把 DDP、TP、CP、EP、MoE 路由都映射回几种基本 collective。

## Reader Baseline
- 读者知道什么是多卡训练，但不一定能清楚区分常见 collective 的语义。
- 不要求读者先懂 NCCL 的具体实现算法。

## Core Questions
- all-reduce、all-gather、reduce-scatter、all-to-all 分别是谁给谁、谁拿到完整结果？
- 为什么 reduce-scatter + all-gather 常常是重要组合？
- 参数同步、激活重分布、专家路由分别对应哪些 collective？
- 什么叫“语义正确但性能错误”的通信？

## Section Arc
1. 开场问题：多卡训练里，到底是谁在拿完整张量，谁在拿分片。
2. 先用 4 卡纸面例子讲所有权变化，不先讲 ring / tree。
3. all-reduce：每个人都拿到归约后的完整结果。
4. all-gather 与 reduce-scatter：完整与分片之间的来回转换。
5. all-to-all：拥有者和消费者一起改变，是 MoE 等系统问题的核心动作。
6. 把 DDP、TP、CP、EP、MoE 路由映射回 collective 组合。
7. 结尾收束：通信问题先是语义问题，后才是实现问题。

## Claims Requiring Current Verification
- 如果正文要写 NCCL 某一版本的行为、限制或调优建议，需要按当时官方文档核验。
- 如果正文要把 differentiable collectives 或其他新接口放进“2026 侧栏”，需要按当时 PyTorch/NCCL 文档核验。

## Examples / Figures / Comparisons
- 贯穿例子：4 卡上的参数同步、序列切分、专家路由。
- 默认用文字和小表格解释，不主动画图。
- 如确实需要图，可搜：`NCCL collective operations official`。

## Writing Guardrails
- 本章先稳住语义，不急着进入 ring、tree、topology 细节。
- 先解释“张量怎么变”，再解释“为什么会快/慢”。
- 首次出现就定义：rank、collective、shard、replica、owner。

## Open Questions for User
- 这章你更想停在 collective 语义，还是顺带补一点 ring / tree / topology 的性能直觉？
