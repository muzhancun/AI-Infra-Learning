"""
Chapter 6 exercise 1: read timeline gaps before chasing kernel math.

Scenario
--------
Imagine one training or inference step contains the following work:

- one thicker matmul
- several tiny pointwise-ish ops
- one layernorm

On paper, the tiny ops look cheap. On a timeline, they may not be cheap at
all, because each one can pay its own launch/host overhead. This toy script
lets you compare three execution styles:

1. eager: each op launches separately
2. fused: several tiny ops are merged into one larger group
3. CUDA-graph-like replay: launch/CPU overhead per group becomes smaller

The goal is not exact modeling. The goal is to train one chapter habit:

    "When a workload looks fragmented, how much time is going into
     the kernels themselves, and how much is going into the gaps?"

Run
---
python "AI Infra/硬件、运行时与 profiling/Chap6/scripts/timeline_gaps.py"
"""

from dataclasses import dataclass


@dataclass
class Op:
    name: str
    kernel_us: float
    launch_us: float = 8.0


OPS = [
    Op("matmul", 120.0),
    Op("bias_add", 6.0),
    Op("relu", 4.0),
    Op("dropout", 7.0),
    Op("layernorm", 16.0),
]

# A toy compile pass fused the 3 pointwise-ish ops into one group.
FUSED_GROUPS = [
    [OPS[0]],
    [OPS[1], OPS[2], OPS[3]],
    [OPS[4]],
]


def kernel_only_total_us(ops: list[Op]) -> float:
    """
    Pure device-side compute time in this toy world.

    This is the "no gaps at all" reference point. Real systems never quite
    reach it, but it helps us separate actual kernel work from launch/host
    overhead.
    """
    return sum(op.kernel_us for op in ops)


def eager_total_us(ops: list[Op]) -> float:
    """TODO 1: in eager mode, each op pays its own launch cost."""
    raise NotImplementedError("Fill TODO 1")


def fused_total_us(groups: list[list[Op]]) -> float:
    """TODO 2: in fused mode, each group pays one launch cost."""
    raise NotImplementedError("Fill TODO 2")


def cudagraph_replay_total_us(groups: list[list[Op]], replay_overhead_us: float = 2.0) -> float:
    """TODO 3: pretend each captured group replays with smaller CPU/launch overhead."""
    raise NotImplementedError("Fill TODO 3")


def speedup(old_us: float, new_us: float) -> float:
    """TODO 4: compute old/new speedup."""
    raise NotImplementedError("Fill TODO 4")


def main() -> None:
    print("Scenario: compare one fragmented eager step with fused execution")
    print("and CUDA-graph-like replay. The question is not just 'how much")
    print("kernel time exists', but also 'how much timeline gap do we pay'.")
    print()

    kernel_only = kernel_only_total_us(OPS)
    eager = eager_total_us(OPS)
    fused = fused_total_us(FUSED_GROUPS)
    replay = cudagraph_replay_total_us(FUSED_GROUPS)

    print(f"kernel-only   : {kernel_only:8.2f} us  (toy lower bound: no launch gaps)")
    print(f"eager total   : {eager:8.2f} us")
    print(f"fused total   : {fused:8.2f} us")
    print(f"graph replay  : {replay:8.2f} us")
    print(f"eager overhead: {eager - kernel_only:8.2f} us")
    print(f"fused overhead: {fused - kernel_only:8.2f} us")
    print(f"replay overhd.: {replay - kernel_only:8.2f} us")
    print(f"fused speedup : {speedup(eager, fused):8.3f}x")
    print(f"replay speedup: {speedup(eager, replay):8.3f}x")

    print()
    print("思考题：")
    print("1) 为什么最短的小 kernel 最容易被 launch overhead 吃掉？")
    print("2) 这里哪种优化更像解决 kernel 过碎，哪种更像进一步压缩 Python/host 提交开销？")
    print("3) 如果 shape 频繁变化，这类收益为什么会更不稳定？")


if __name__ == "__main__":
    main()
