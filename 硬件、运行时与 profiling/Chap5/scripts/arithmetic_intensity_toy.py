"""
Chapter 5 exercise 1: arithmetic intensity as a first debugging lens.

Scenario
--------
Imagine you are reading a small-batch inference trace. You see matmul,
ReLU, layernorm, and embedding lookup all appear in the timeline.
Some kernels are short but show up again and again. Others are thick,
dense chunks of work. Before you rush to optimize anything, you want a
very rough answer to one question:

    "Which ops are doing a lot of math per byte moved,
     and which ops are mostly paying for data movement?"

This script is a toy calculator for that first-pass judgment.

What this exercise is for
-------------------------
- It is NOT an exact performance model.
- It WILL help you build the habit of comparing "compute work" and
  "data movement" before talking about optimization.
- The goal is to classify the *personality* of an op:
  more compute-heavy, or more memory-heavy.

Run
---
python "AI Infra/硬件、运行时与 profiling/Chap5/scripts/arithmetic_intensity_toy.py"
"""

BYTES_PER_ELEM = 2  # toy counting: think fp16 / bf16-ish tensors


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """
    FLOPs/Byte: how much arithmetic work we do for each byte moved.

    Higher arithmetic intensity usually means the op has a better chance
    to behave like a compute-heavy workload.
    Lower arithmetic intensity usually means data movement is more likely
    to dominate.
    """
    return flops / bytes_moved if bytes_moved else 0.0


def matmul_counts(m: int, n: int, k: int, bytes_per_elem: int = BYTES_PER_ELEM):
    # Toy model:
    # A is [m, k], B is [k, n], C is [m, n].
    # FLOPs use the classic GEMM approximation: 2 * m * n * k.
    #
    # For bytes, we use an intentionally optimistic picture:
    # - read A once
    # - read B once
    # - write C once
    #
    # Real kernels are more complicated, but this is enough to build the
    # intuition that matmul can often reuse data and achieve high AI.
    flops = 2 * m * n * k
    # TODO 1: count bytes if we read A once, read B once, and write C once.
    raise NotImplementedError("Fill TODO 1")


def relu_counts(numel: int, bytes_per_elem: int = BYTES_PER_ELEM):
    # Toy model:
    # Each element does only a tiny amount of math, roughly one compare
    # or max-like operation, but every element still has to be read and
    # the result written back.
    flops = numel
    # TODO 2: read x once, write y once.
    raise NotImplementedError("Fill TODO 2")


def layernorm_counts(tokens: int, hidden: int, bytes_per_elem: int = BYTES_PER_ELEM):
    # Toy model:
    # We flatten the tensor into tokens * hidden elements.
    # FLOPs are kept rough on purpose: mean, variance, normalize,
    # scale, bias.
    #
    # The point here is not exact accounting. The point is to notice that
    # layernorm usually touches a lot of data relative to how much math it
    # performs, so it often behaves more like a memory-bound op.
    numel = tokens * hidden
    flops = 5 * numel  # toy count: mean, variance, normalize, scale, bias
    # TODO 3: give a rough byte count. At minimum, think about reading x,
    # reading gamma/beta, and writing y once each.
    raise NotImplementedError("Fill TODO 3")


def embedding_lookup_counts(tokens: int, hidden: int, bytes_per_elem: int = BYTES_PER_ELEM):
    # Toy model:
    # We ignore arithmetic almost entirely and focus on what this op often
    # feels like in practice: gather rows from a table, then write them out.
    #
    # That is exactly why embedding lookup is a good chapter example:
    # it can be slow even though it does very little "math" in the usual
    # sense.
    flops = 0
    # TODO 4: read selected rows once and write the gathered output once.
    raise NotImplementedError("Fill TODO 4")


def show(name: str, flops: float, bytes_moved: float) -> None:
    ai = arithmetic_intensity(flops, bytes_moved)
    print(f"{name:16s} FLOPs={flops:12.0f}  Bytes={bytes_moved:12.0f}  FLOPs/Byte={ai:8.2f}")


def main() -> None:
    print("Scenario: you are reading a small-batch inference trace.")
    print("Use toy FLOPs/Byte estimates to judge which ops look compute-heavy")
    print("and which ones look more like data-movement problems.")
    print()

    m, n, k = 4096, 4096, 4096
    show("matmul", *matmul_counts(m, n, k))
    show("relu", *relu_counts(4_096 * 4_096))
    show("layernorm", *layernorm_counts(tokens=1024, hidden=4096))
    show("embedding", *embedding_lookup_counts(tokens=1024, hidden=4096))

    print()
    print("思考题：")
    print("1) 在这个 toy trace 里，哪些 op 最像“搬得多、算得少”，因此更值得先怀疑 memory-bound？")
    print("2) 如果 batch 变小、tile 变差，为什么 matmul 也可能没有原来那么“算力型”？")
    print("3) 为什么 embedding lookup 看起来几乎没有 FLOPs，却仍然可能在 profiler 里很慢？")


if __name__ == "__main__":
    main()
