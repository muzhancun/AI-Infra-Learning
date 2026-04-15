"""
Answer for arithmetic_intensity_toy.py.

The formulas below stay deliberately simple. Their job is not to predict
real hardware performance exactly, but to support the chapter habit:
compare math work and data movement first, then judge whether an op feels
compute-heavy or memory-heavy.
"""
BYTES_PER_ELEM = 2

def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    return flops / bytes_moved if bytes_moved else 0.0

def matmul_counts(m: int, n: int, k: int, bytes_per_elem: int = BYTES_PER_ELEM):
    # Read A once, read B once, write C once.
    flops = 2 * m * n * k
    bytes_moved = (m * k + k * n + m * n) * bytes_per_elem
    return flops, bytes_moved

def relu_counts(numel: int, bytes_per_elem: int = BYTES_PER_ELEM):
    # Read input once, write output once.
    flops = numel
    bytes_moved = 2 * numel * bytes_per_elem
    return flops, bytes_moved

def layernorm_counts(tokens: int, hidden: int, bytes_per_elem: int = BYTES_PER_ELEM):
    # Roughly: read x, read gamma/beta, write y.
    numel = tokens * hidden
    flops = 5 * numel
    bytes_moved = (numel + numel + 2 * hidden) * bytes_per_elem
    return flops, bytes_moved

def embedding_lookup_counts(tokens: int, hidden: int, bytes_per_elem: int = BYTES_PER_ELEM):
    # Gather selected rows once, then write gathered output once.
    flops = 0
    bytes_moved = 2 * tokens * hidden * bytes_per_elem
    return flops, bytes_moved

def show(name: str, flops: float, bytes_moved: float) -> None:
    ai = arithmetic_intensity(flops, bytes_moved)
    print(f"{name:16s} FLOPs={flops:12.0f}  Bytes={bytes_moved:12.0f}  FLOPs/Byte={ai:8.2f}")

def main() -> None:
    print("Scenario: classify op personality before deeper profiling.")
    print()
    m, n, k = 4096, 4096, 4096
    show("matmul", *matmul_counts(m, n, k))
    show("relu", *relu_counts(4_096 * 4_096))
    show("layernorm", *layernorm_counts(tokens=1024, hidden=4096))
    show("embedding", *embedding_lookup_counts(tokens=1024, hidden=4096))

if __name__ == "__main__":
    main()
