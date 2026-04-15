# 第四章：GPU 不是黑盒

你可以先把 GPU 想成一个只认 kernel 的机器。它不认识“这一层是 attention”“那一层是 layernorm”，也不关心模型名字。它真正接到手里的，是一串被发射出去的 kernel：每个 kernel 负责一段固定的计算，带着自己的输入、输出和资源需求，在有限的硬件上争取尽快完成。

这件事听起来很朴素，但它会立刻改变我们看性能的方式。很多人第一次看 GPU 性能，只会说“它没跑满”。可“没跑满”并不是解释，它只是一个现象。真正该问的是：是 kernel 太小，还是内存太慢，还是并发被资源约束卡住了，还是 host 端根本没有把工作连续地喂给 GPU。

这一章要做的，就是把 GPU 从黑盒里拆出来，先建立一个最小但足够用的机器模型。

## 从 op 到 kernel

在 PyTorch 里，你看到的是 op；在 GPU 上，真正执行的是 kernel。op 是语义层上的动作，比如 matmul、layernorm、relu；kernel 是实现层上的动作，是某一段具体的代码，按照线程划分、内存访问方式和同步方式去完成这件事。一个 op 不一定对应一个 kernel，一个 kernel 也不一定只服务一个 op。编译器、fusion、框架调度和底层实现，都会影响最后到底发出去多少个 kernel。

这就是为什么同样是“一个层”，在 profiler 里看到的形状可以完全不同。有的层像一块厚实的工作，kernel 少、时间连续、device 忙得比较均匀；有的层则像很多碎片化的小活，被切成一串短 kernel，中间夹着等待和空洞。前者不一定更“高级”，后者也不一定更“笨”，只是它们对硬件的使用方式不同。

> [!abstract] 定义：kernel
> kernel 是一次被 runtime 或框架发射到 GPU 上执行的设备端程序。它按 thread、block、grid 的组织方式展开，负责完成一段具体计算，是 GPU 实际执行的基本工作单位。
> 在性能分析里，kernel 比 op 更靠近硬件，因此往往也是更直接的观察对象。

> [!abstract] 先记住一个最基本的区分
> op 是你在模型层描述的事情，kernel 是 GPU 真正在做的事情。
> 讨论性能时，op 的名字只能提供线索，kernel 的形状才是第一手证据。

## profiler 在看什么

profiler 可以先理解成一种“执行过程记录仪”。它不替你优化程序，也不直接告诉你哪里一定有问题；它做的事，是把 CPU 和 GPU 在一段时间里到底执行了什么、谁先谁后、每段工作花了多久、有没有等待、有没有空洞，尽量还原出来。

如果说 trace 是“发生过什么”的时间线，那么 profiler 就是生成这条时间线、并附带统计信息的工具。它通常会记录 op、kernel、CPU 调用、CUDA API、memory copy、同步等待等事件，并把它们放到统一的时间轴上。这样你看到的就不只是“慢”，而是“慢在 host 没发出活、慢在 kernel 太碎、慢在 memcpy、还是慢在某个 kernel 自己”。

所以更准确地说，profiler 不是在看“模型语义”，而是在看“执行事实”。模型里你写的是 layer、attention、loss；profiler 里你看到的是 op、kernel、launch、memcpy、sync，以及这些事件在时间线上如何拼起来。也正因为如此，profiler 是理解性能问题时最接近现场的工具之一。

> [!abstract] 定义：profiler
> profiler 是用来采集、记录并展示程序执行行为的工具。
> 在 GPU 场景里，它通常把 op、kernel、CUDA API、memory copy、同步等待等事件放到时间轴和统计视图里，帮助我们定位时间到底花在了哪里。

## SM、warp、block 是什么

GPU 不是一块整体同时工作的晶体，而是由很多个可以并发执行的小单位组成。这里最重要的几个词是 SM、grid、block、warp 和 thread。

**SM** 是 streaming multiprocessor，可以把它理解成 GPU 里负责真正执行计算的核心单元。**thread** 是最细的逻辑执行单元。**block** 是 thread 的组织单位，一个 kernel launch 会产生很多 block。**grid** 则是这次 launch 的全部 block 合在一起形成的整体。**warp** 不是你额外手写出来的一层任务，而是硬件把同一个 block 里的线程按固定大小分组后形成的执行分组；在本章的 CUDA 语境里，可以先把它理解成通常由 32 个线程组成的一队人。block 里的线程彼此可以协作、共享局部资源，但不同 block 之间一般要假定彼此独立，因为它们可能被分配到不同 SM，运行顺序也不可靠。CUDA C++ Programming Guide 也明确要求 thread block 必须能够独立执行，这样同一个 kernel 才能透明扩展到任意数量的 multiprocessor。([CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf))

这三个概念的关系很重要，因为它决定了 GPU 的并发方式不是“想跑多少就跑多少”，而是“在 block 和 warp 的组织下，尽量让 SM 持续有活可干”。如果一个 kernel 的 block 太少，SM 就可能闲下来；如果每个 block 占用的资源太大，SM 也装不下太多 block；如果线程之间的工作方式不适合 warp 的执行模型，执行效率也会掉下来。

> [!abstract] 定义：SM、grid、block、warp、thread
> SM：GPU 中负责执行计算与调度执行资源的核心单元。
> grid：一次 kernel launch 产生的全部 block 的集合。
> block：kernel 组织出来的线程块，block 之间通常相互独立。
> thread：block 内最细的逻辑线程。
> warp：硬件把同一个 block 内的线程按固定大小分成的小组；在本章里可先按 32 个线程一组理解，它通常是调度与执行的基本粒度之一。

你可以把它们想成两套同时成立的视角：从“程序怎么组织”看，是 **kernel → grid → block → thread**；从“硬件怎么执行”看，是 **SM 取走一个个 block，再按 warp 去推进 block 里的线程**。只要这个组织方式不合适，GPU 就算有很多计算资源，也未必能把它们用满。

> [!tip] kernel、block、warp、thread 的关系
> 更准确的层级关系是：
> **一次 kernel launch 会产生一个 grid；grid 里有很多 block；每个 block 里有很多 thread；这些 thread 会再按 warp 大小被硬件分组执行。**
> 所以常见的理解可以写成：
> **1 kernel → K 个 block；每个 block → T 个 thread；每个 block 大约有 ceil(T / 32) 个 warp。**
> 例如一个 kernel 如果有 100 个 block，而每个 block 有 256 个线程，那么每个 block 对应 8 个 warp，整个 kernel 一共会形成 25600 个 thread 和 800 个逻辑 warp。之后调度器再把这些 block 分配到不同 SM 上执行。

> [!tip] 一个最容易混的点
> **warp 不是跑在 block 外面的一层新任务。**
> 更准确地说，warp 是 **block 内线程的执行分组**，而且 warp 不会跨 block 拼接。
> 所以 block 更像“怎么分活”，warp 更像“这些活在硬件上怎么被一组一组地跑起来”。

> [!tip] 另一个容易忽略的点
> **一个 kernel 里的全部 block 通常不会同时跑完。**
> kernel 可以有很多 block，但一个 SM 同时能驻留多少 block，要受线程数、register、shared memory、warp 槽位等资源限制。放不下的 block 会排队，等前一批执行完再进入下一 wave。

## 现实插页：研究者会在 Triton kernel 里看到什么

如果你去看 `flash-linear-attention` 这类 Triton-heavy 项目里的 kernel，真正看到的通常不是“attention”这个词本身，而是一段更像下面这样的代码骨架：

```python
import triton
import triton.language as tl


# @triton.jit 表示：下面这个 Python 函数会被 Triton 编译成 GPU kernel。
# 也就是说，这不是普通的 host 端 Python 逻辑，而是 device 端要执行的程序。
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # tl.constexpr 表示这个参数在编译期已知；Triton 可以据此做更激进的展开和优化。

    # program_id 可以先理解成“这一个程序实例 / 线程块正在处理第几块数据”。
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK

    # tl.arange 会生成当前这块里的一串局部 offsets。
    offsets = block_start + tl.arange(0, BLOCK)

    # mask 用来保护边界：最后一块可能不满，越界位置就不应该真的访问内存。
    mask = offsets < n_elements

    # tl.load / tl.store 是 Triton kernel 最常见的访存原语。
    # 它们看起来像“从指针读 / 往指针写”，本质上就是这块 kernel 在搬数据。
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


# grid 决定这次 launch 一共发出去多少个程序实例。
# triton.cdiv 是向上取整除法；数据块不整除时，最后一块仍然要发出去。
grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)

# kernel[grid](...) 是 Triton 的 launch 写法：按这个 grid 把 kernel 发到 GPU 上执行。
add_kernel[grid](x, y, out, n_elements, BLOCK=1024)
```

这不是 `fla` 里的某个完整 kernel，而是把那类代码最常见的骨架压缩成一个最小例子。真正的 FLA kernel 往往会更长：参数里会有 `q`/`k`/`v` 指针、stride、`BLOCK_M` / `BLOCK_N` / `BLOCK_D`，以及更复杂的 `tl.load`、`tl.store` 和局部累加。但你现在至少能先认出几件事。`@triton.jit` 定义的是 device 端 kernel；`pid` 对应的是当前程序实例负责哪一块数据；`BLOCK` 这类 `constexpr` 参数决定 tile 大小；`grid` 决定这次 launch 一共发出去多少块。换句话说，你在真实项目里看到的一堆指针、stride 和 block 参数，本质上还是这一章前面那套 kernel、grid、block、thread 的故事，只是从讲义语言变成了代码语言。

## global memory 和 on-chip memory 不是同一类东西

真正让 GPU 性能分叉的，常常不是“算得快不快”，而是“数据搬得快不快”。GPU 上最重要的一条分界线，是 global memory 和 on-chip memory。

global memory 指的是 GPU 直接访问的外部显存。它容量大，但访问代价高。on-chip memory 则是芯片内部更快、更近的那部分资源，通常包括 register、shared memory 以及相关缓存层次。register 最快也最稀缺，shared memory 适合 block 内协作，global memory 则承担大规模存储。这和 CUDA Programming Guide 以及 Best Practices 的表述一致：shared memory 属于 on-chip memory，而 global memory 对应更远、更贵的外部存储层。([CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf), [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf))

这三者的关系，不是简单地“越快越好”，而是每一层都有自己的角色。计算如果每次都去碰最慢的那层，时间就会花在等待数据回来上；如果把常用数据放在更近的层里，kernel 就能把更多时间用在真正计算上。很多优化的本质，其实就是把数据尽量留在离计算更近的地方，并减少来回搬运。

layernorm 和 relu 这类操作特别适合说明这一点。它们的计算并不复杂，但它们经常要读很多数据、写很多数据，计算量和搬运量之间的比例不高，所以性能往往更容易被内存访问拖住。matmul 则常常更像另一类情况：它有更高的计算密度，能更充分地把计算单元喂起来，因此在很多场景里更接近算力受限。

> [!tip] 一个很实用的直觉
> 看到慢的算子时，先不要急着问“它算得怎么样”，先问“它的数据是怎么放的、怎么搬的、能不能重复利用”。

## 为什么 register、shared memory、block size 会一起限制并发

GPU 并发的上限，不是一个单一数字，而是几种资源一起决定的。一个 block 运行时，会占用一定数量的 register，会占用一定数量的 shared memory，也会需要一定数量的线程和 warp 位置。只要这些资源中的某一种先满了，SM 就不能再随意塞进更多 block。

这就是为什么 block size 不是越大越好，register 用得越多也不一定越好。block 太大，可能让一个 SM 同时能驻留的 block 变少；register 用得太豪华，可能让每个线程“住得很舒服”，但总并发下降；shared memory 如果被占得太满，block 之间就更难并行起来。GPU 的执行方式很像一种资源分配问题：每个 kernel 都在争夺有限的床位，而不是无限扩张。

这里常见的一个词是 occupancy。它可以简单理解成：一个 SM 当前能同时活跃的 warp 数量，相对于它理论最大可活跃 warp 数量的比例。occupancy 高，通常说明硬件上有更多线程在等着执行，发生延迟时更容易用别的 warp 去遮住；occupancy 太低，说明很多执行槽位空着，延迟更难被隐藏。

但 occupancy 不是性能的结论。它只是一个中间指标。高 occupancy 不代表一定快，因为如果每个线程都在做很差的内存访问，或者 kernel 本身就被带宽卡住了，更多活跃 warp 也不一定能解决问题。反过来，有些 kernel 即使 occupancy 不高，也可能因为计算非常规整、数据复用很好、整体访问很紧凑而表现不错。

> [!abstract] 定义：occupancy
> occupancy 是活跃 warp 占理论最大活跃 warp 的比例。
> 它常常影响 latency hiding，但它不是性能本身。
> 看到 occupancy，应该继续问资源是否受限、访存是否规整、数据复用是否充分。

这和 NVIDIA 官方文档的表述是一致的：Best Practices 把 occupancy 定义成 active warps 相对最大 possible active warps 的比例，并明确提醒它首先影响 latency hiding，而更高 occupancy 并不自动等于更高性能。([CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf))

所以更稳妥的说法是：occupancy 提供的是“这块 SM 还有多少并发余地”的线索，而不是“这段 kernel 到底快不快”的结论。要判断后者，还得看内存访问、数据复用、指令组合和 kernel 的整体结构。

## 用 matmul、layernorm、relu 看三种不同的气质

如果把 matmul、layernorm、relu 放在一起看，你会很快发现它们的气质不同。

matmul 往往更像一个“把计算堆起来”的算子。它会把很多乘加组织成连续的工作，容易形成比较稳定的计算密度，因此在很多情况下更接近算力受限。layernorm 则更像“在读写中计算”，它要对一段向量做归一化，计算本身不重，但数据流动很多，性能常常被 memory traffic 牵着走。relu 甚至更极端，它看起来几乎只是做一次逐元素判断，但正因为计算太轻，真正耗时的地方常常变成读内存和写内存。

这三者放在一起，能帮我们建立一个很关键的判断：慢不一定是因为“算不动”，也可能是因为“搬不动”或“组织得太碎”。matmul 慢，和 relu 慢，未必是同一种慢。前者可能要看计算单元和数据复用，后者往往更该看 memory bandwidth、kernel launch 和并发组织。

如果你以后在 trace 里看到一个 op 很慢，不要急着把它等同于“模型大”。先看它属于哪一类气质：它更像厚实的计算块，还是更像大量细碎的数据搬运。这个判断，往往比盯着名字本身更有用。

## 结尾：GPU 不是黑盒，而是一套受限的执行秩序

把这一章收束成一句话，可以这样说：GPU 不是一块神秘的加速卡，而是一套围绕 kernel、SM、warp、block 和 memory hierarchy 组织起来的受限执行秩序。

它的快，不是因为它“更强”，而是因为它能把大量并发、规整访问和有限资源调度得更好。它的慢，也不只是因为某个算子“不够优化”，而是因为执行被资源、访存和组织方式共同塑形了。occupancy、block size、register、shared memory 这些词，本质上都在描述同一件事：这段工作能不能被稳定而高效地铺到硬件上。

下一章我们会继续往下走，把“硬件上能跑”进一步推进到“为什么很多时间不是花在算，而是花在搬”。一旦你开始用这种方式看 GPU，很多原本模糊的性能问题就会变得具体起来：到底是线程不够，还是资源被占满，还是内存没喂饱，还是 kernel 太碎。真正理解 GPU，从来不是背名词，而是学会把一张 trace 读成一套执行机制。

## 代码练习
- 练习 1：[[AI Infra/硬件、运行时与 profiling/Chap4/scripts/occupancy_toy.py|occupancy_toy.py]]；答案 [[AI Infra/硬件、运行时与 profiling/Chap4/scripts/occupancy_toy_answer.py|occupancy_toy_answer.py]]
- 练习 2：[[AI Infra/硬件、运行时与 profiling/Chap4/scripts/block_scheduler_toy.py|block_scheduler_toy.py]]；答案 [[AI Infra/硬件、运行时与 profiling/Chap4/scripts/block_scheduler_toy_answer.py|block_scheduler_toy_answer.py]]
- 索引：[[AI Infra/硬件、运行时与 profiling/Chap4/scripts/README]]

## 思考题答案

### 练习 1：occupancy toy

参考答案运行后，可以得到这样一组结果：`matmul_like` 的 occupancy 是 50%，`layernorm_like` 和 `relu_like` 都是 100%。

**1）哪个 kernel 更可能先被寄存器限制？**

`matmul_like` 更可能先被寄存器限制。因为它每个线程要用 64 个寄存器，高于另外两个例子。寄存器预算固定时，每个 block 越“吃寄存器”，一个 SM 同时能容纳的 block 就越少。

**2）哪个 kernel 更可能 occupancy 高，但仍然未必最快？**

`relu_like` 是最典型的例子。它的 occupancy 很高，但逐元素计算通常很轻，真正的瓶颈往往不在算力，而在 memory bandwidth、kernel launch 开销和整体调度碎片化。所以 occupancy 高，并不自动等于它最快。

**3）如果把 `matmul_like` 的 `threads_per_block` 从 256 改成 512，会发生什么？**

每个 block 的线程数翻倍后，单个 block 会包含更多 warp，但一个 SM 同时能容纳的 block 数会下降。对这个 toy 例子来说，`blocks/SM` 会从 4 变成 2，而 `warps/block` 会从 8 变成 16，所以最后 `active warps/SM` 仍然是 32，occupancy 还是 50%。这说明 block size 改变后，occupancy 不一定跟着改变，但调度粒度和资源分配方式已经变了。

### 练习 2：block scheduler toy

参考答案运行后，可以看到：当 block 总数少于 SM 数量时，会有一部分 SM 根本分不到工作；当 block 足够多时，所有 SM 都能先忙起来；如果 block 再继续增加，就会进入多 wave 执行。

**1）为什么 block 必须彼此独立，调度器才能放心地把它们分给不同 SM？**

因为 block 的落点和先后顺序并不可靠。某个 block 可能先运行，也可能后运行；可能在这个 SM 上，也可能在另一个 SM 上。只有 block 彼此独立，调度器才能自由分配，而不用担心跨 block 依赖把执行顺序卡死。

**2）为什么 block 太少时，GPU 可能还没忙起来？**

因为 GPU 有很多 SM，但如果 block 数量太少，就不是每个 SM 都能分到工作。比如 8 个 SM 只有 4 个 block，那么最多也就只有 4 个 SM 在忙，剩下的 SM 会空着。

**3）为什么 occupancy 和 block 总数都只是线索，不是最后的性能答案？**

因为它们主要告诉我们“并发空间够不够”，却没有直接回答访存是否规整、数据复用是否充分、kernel launch 是否频繁、同步等待是否严重。occupancy 高只能说明可能更容易隐藏延迟，block 多只能说明工作被切得足够开；真正快不快，还要看 kernel 的访存模式和整体执行结构。

## 本章 API 速记

- `@triton.jit`：把一个 Python 函数标记成 Triton kernel，让它被编译成 GPU 上执行的程序。
- `tl.constexpr`：声明某个参数在编译期已知，常用于 block / tile 大小这类配置。
- `tl.program_id(axis=0)`：当前 Triton 程序实例在某个维度上的编号；可以先把它理解成“当前这块在处理第几段数据”。
- `tl.arange(start, end)`：在 kernel 内生成一段局部连续索引，常用来构造当前 block 的 offsets。
- `tl.load(...)` / `tl.store(...)`：Triton kernel 里最常见的读写原语，用来从显存或其他地址空间读写数据。
- `triton.cdiv(a, b)`：向上取整除法，常用来计算 launch 需要多少个 block / program instance。
- `kernel[grid](...)`：Triton 的 launch 写法；含义是“按这个 grid 把 kernel 发到 GPU 上执行”。

## 本章引用与延伸阅读
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)
- [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [flash-linear-attention — GitHub](https://github.com/fla-org/flash-linear-attention)
