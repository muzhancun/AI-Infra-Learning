下面是一版可直接进入写作的第 1 章设计。它和整本书的定位一致：读者已经懂模型、优化和 RL，但还没有把“大模型系统”当成独立研究对象；因此这一章不按框架名组织，而按系统对象与系统瓶颈组织。到 2026，训练侧的主线确实已经收敛到 FSDP2、DCP、torch.compile、FlexAttention、Megatron Core，推理侧围绕 KV cache、prefix caching、continuous batching、disaggregated serving，RL 后训练侧越来越像分布式数据流系统。

我建议把这一章的总命题写成一句话：**AI Infra 研究的不是“层长什么样”，而是“状态在什么时间、驻留在哪、如何移动、由谁调度，以及这些选择怎样决定吞吐、延迟、稳定性和成本”。** 这个表述和 2026 年主流系统栈是对齐的：FSDP2把参数、梯度、优化器状态当作可分片对象；Megatron Core 的 context parallelism 直接按序列维切输入和激活；vLLM 与 NVIDIA Dynamo 把 KV cache、排队、路由、prefill/decode 分离、KV 转移和缓存感知路由放到系统中心。

这一章可以分成六节。

第一节叫“从模型对象到系统对象”。核心任务是把读者从“我在研究网络结构”切到“我在研究状态流”。这里先给出全章的七个对象：参数、激活、优化器状态、KV cache、网络流量、调度、时间。参数决定模型能否放下，激活决定长序列和大 batch 是否可行，优化器状态决定训练显存与 checkpoint 体积，KV cache 决定 serving 的内存经济学，网络流量决定多卡多机扩展性，调度决定资源是否空转，时间则把这一切统一到 step、request、episode 三条时间轴上。

第二节叫“训练时真正占内存和时间的东西”。这里不要讲 Transformer 细节，而是讲一个训练 step 的状态账本：参数、梯度、优化器状态、激活、临时 buffer、通信 staging、checkpoint I/O。用 FSDP2 引出“参数/梯度/优化器状态是可以被切分和重组的”，用 context parallel 引出“激活也可以沿序列维切开”，再用 DCP 引出“checkpoint 不是训练后的附属品，而是训练路径上的 I/O 与恢复对象”。FSDP2 的官方教程和文档都直接把参数、梯度、优化器状态的分片作为核心；DCP 解决的是多 rank 并行保存/加载以及跨拓扑 resharding；async save 的意义则是把 checkpoint 尽量移出 critical path。

第三节叫“推理系统为什么围着 KV cache 转”。这一节要让读者意识到：训练阶段的主角是参数与激活，推理阶段的主角往往是 KV cache。vLLM 把 PagedAttention、continuous batching、automatic prefix caching、GPU cache usage、running/waiting requests、prompt tokens/s、generation tokens/s、prefix cache hit rate 都放进了标准指标体系；NVIDIA Dynamo 则进一步把 serving 拆成 prefill engine 生成 KV、把 KV 传给 decode engine、再由 decode engine 继续生成，并围绕 TTFT、ITL、KV cache utilization、KV-aware routing 和 KV offloading 做调度。

第四节叫“AI Infra 关心的指标地图”，这应该是全章最重要的一节。训练侧最该先讲的是 **MFU、step time、tokens/s、峰值显存、checkpoint save/load/restart 开销、graph break 与编译命中情况**。Megatron/NVIDIA 现在仍把 MFU 作为训练效率的头号指标之一，Megatron-LM 公开 benchmark 也仍然用 MFU 汇报大规模训练效率；PyTorch 的 `torch.compile` 文档则把 profiling、graph breaks、compiled region、CUDAGraph Trees 当成性能诊断入口；DCP 文档和博客明确强调 collective communication 与 checkpoint latency 会随着规模上升而成为显著成本。

推理侧最该先讲的是 **TTFT、TPOT/ITL、E2E latency、running/waiting requests、queue interval、prefill interval、prompt tokens/s、generation tokens/s、GPU/KV cache utilization、prefix cache hit rate，以及在 SLA 约束下能撑住的最大并发**。vLLM 的基准与监控接口已经把 `ttft`、`tpot`、`itl`、`e2el` 作为标准 percentile 指标，日志里也直接输出 running/waiting、GPU cache usage、prompt tok/s、new tok/s、prefix cache hit rate；Dynamo Planner 则明确说明 LLM 服务的 autoscaling 目标不再是 CPU 利用率这类通用云指标，而是 TTFT、ITL 和 KV cache utilization 这类 LLM 原生 SLA 指标。

RL 后训练这一节不要先讲 PPO/GRPO 数学，而要讲 **rollout 和 update 能否重叠、资源是否空转、队列是否堆积、版本同步是否及时、checkpoint 是否足够稳**。TRL 现在已经把理解日志、减显存、提速和分布式训练做成独立文档面向用户；OpenRLHF 明确把自己定义成基于 Ray + vLLM 的可扩展 RLHF 框架，并提供 hybrid engine 来减少 GPU idle；verl 则把 HybridFlow/fully async trainer-rollouter 作为系统特性来讲，公开文档里直接给出异步 trainer/rollouter 解耦以及 128 GPU 上 2.35x–2.67x 的性能提升；Ray 的调度和 autoscaling 文档也说明系统瓶颈常常表现为队列、资源放置和扩缩容决策。

第五节叫“这章要反复纠正的三个误解”。第一，**GPU utilization 不是全部**，训练要看 MFU，服务要看 TTFT/ITL，RL 要看 rollout-train overlap。第二，**显存不是一个总数，而是不同状态对象的政治版图**，参数、激活、优化器状态、KV cache 会争抢同一块 HBM。第三，**性能不是 kernel 的局部属性，而是调度与数据流的全局属性**：queue、prefill/decode 切分、cache hit、跨节点 KV 传输、checkpoint collective 都会反过来决定“看上去像模型问题”的速度问题。

第六节是本章结论，建议收束成一句定义：**AI Infra 的核心，就是在显存、带宽、延迟和可靠性约束下，让“有用 token”连续地流过系统。** 训练里你用 MFU、step time 和 checkpoint 开销来衡量它；推理里你用 TTFT、ITL/TPOT、cache hit rate 和 SLA 下最大并发来衡量它；RL 后训练里你用 rollout 与学习是否重叠、资源是否空转、恢复是否稳定来衡量它。

如果你要把这一章真正写成“书稿风格”，最好的开篇句可以是这一句：**“大模型系统并不是一堆 GPU 上跑着一个模型，而是一批状态对象在显存、网络和时间上的受限流动。”**