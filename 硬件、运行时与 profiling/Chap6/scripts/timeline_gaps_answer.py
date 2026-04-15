"""
Answer for timeline_gaps.py.

This answer keeps the arithmetic simple on purpose. The point is to make
the "kernel time vs gap time" split visible.
"""
from dataclasses import dataclass

@dataclass
class Op:
    name: str
    kernel_us: float
    launch_us: float = 8.0

OPS = [Op("matmul", 120.0), Op("bias_add", 6.0), Op("relu", 4.0), Op("dropout", 7.0), Op("layernorm", 16.0)]
FUSED_GROUPS = [[OPS[0]], [OPS[1], OPS[2], OPS[3]], [OPS[4]]]

def kernel_only_total_us(ops: list[Op]) -> float:
    return sum(op.kernel_us for op in ops)

def eager_total_us(ops: list[Op]) -> float:
    return sum(op.kernel_us + op.launch_us for op in ops)

def fused_total_us(groups: list[list[Op]]) -> float:
    total = 0.0
    for group in groups:
        total += sum(op.kernel_us for op in group) + group[0].launch_us
    return total

def cudagraph_replay_total_us(groups: list[list[Op]], replay_overhead_us: float = 2.0) -> float:
    total = 0.0
    for group in groups:
        total += sum(op.kernel_us for op in group) + replay_overhead_us
    return total

def speedup(old_us: float, new_us: float) -> float:
    return old_us / new_us

def main() -> None:
    print("Scenario: compare kernel work with timeline gap overhead.")
    print()
    kernel_only = kernel_only_total_us(OPS)
    eager = eager_total_us(OPS)
    fused = fused_total_us(FUSED_GROUPS)
    replay = cudagraph_replay_total_us(FUSED_GROUPS)
    print(f"kernel-only   : {kernel_only:8.2f} us")
    print(f"eager total   : {eager:8.2f} us")
    print(f"fused total   : {fused:8.2f} us")
    print(f"graph replay  : {replay:8.2f} us")
    print(f"eager overhead: {eager - kernel_only:8.2f} us")
    print(f"fused overhead: {fused - kernel_only:8.2f} us")
    print(f"replay overhd.: {replay - kernel_only:8.2f} us")
    print(f"fused speedup : {speedup(eager, fused):8.3f}x")
    print(f"replay speedup: {speedup(eager, replay):8.3f}x")

if __name__ == "__main__":
    main()
