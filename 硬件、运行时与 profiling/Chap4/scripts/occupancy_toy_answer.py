"""Answer for occupancy_toy.py."""
from dataclasses import dataclass

WARP_SIZE = 32

@dataclass
class SM:
    max_threads: int = 2048
    max_blocks: int = 32
    max_warps: int = 64
    registers_per_sm: int = 65536
    shared_mem_per_sm: int = 100 * 1024

@dataclass
class Kernel:
    name: str
    threads_per_block: int
    registers_per_thread: int
    shared_mem_per_block: int

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def warps_per_block(kernel: Kernel) -> int:
    return ceil_div(kernel.threads_per_block, WARP_SIZE)

def blocks_limited_by_threads(sm: SM, kernel: Kernel) -> int:
    return sm.max_threads // kernel.threads_per_block

def blocks_limited_by_registers(sm: SM, kernel: Kernel) -> int:
    regs_per_block = kernel.threads_per_block * kernel.registers_per_thread
    if regs_per_block == 0:
        return sm.max_blocks
    return sm.registers_per_sm // regs_per_block

def blocks_limited_by_shared_memory(sm: SM, kernel: Kernel) -> int:
    if kernel.shared_mem_per_block == 0:
        return sm.max_blocks
    return sm.shared_mem_per_sm // kernel.shared_mem_per_block

def active_blocks_per_sm(sm: SM, kernel: Kernel) -> int:
    return min(sm.max_blocks, blocks_limited_by_threads(sm, kernel), blocks_limited_by_registers(sm, kernel), blocks_limited_by_shared_memory(sm, kernel))

def active_warps_per_sm(sm: SM, kernel: Kernel) -> int:
    return active_blocks_per_sm(sm, kernel) * warps_per_block(kernel)

def occupancy(sm: SM, kernel: Kernel) -> float:
    return active_warps_per_sm(sm, kernel) / sm.max_warps

def print_report(sm: SM, kernel: Kernel) -> None:
    print(f"{kernel.name:14s} blocks/SM={active_blocks_per_sm(sm, kernel):2d} warps/SM={active_warps_per_sm(sm, kernel):3d} occupancy={occupancy(sm, kernel):5.1%}")

def main() -> None:
    sm = SM()
    kernels = [Kernel("matmul_like", 256, 64, 16 * 1024), Kernel("layernorm_like", 256, 32, 8 * 1024), Kernel("relu_like", 128, 16, 0)]
    for k in kernels:
        print_report(sm, k)

if __name__ == "__main__":
    main()
