"""
Chapter 4 exercise: toy occupancy estimator.

Goal
----
Fill in the TODOs to see how threads/block, registers/thread, and
shared-memory/block can jointly limit active blocks and active warps.

This is intentionally simplified. It is for intuition, not for exact GPU modeling.

Run
---
python "AI Infra/硬件、运行时与 profiling/Chap4/scripts/occupancy_toy.py"
"""

from dataclasses import dataclass

WARP_SIZE = 32


@dataclass
class SM:
    max_threads: int = 2048
    max_blocks: int = 32
    max_warps: int = 64
    registers_per_sm: int = 65536
    shared_mem_per_sm: int = 100 * 1024  # bytes


@dataclass
class Kernel:
    name: str
    threads_per_block: int
    registers_per_thread: int
    shared_mem_per_block: int  # bytes


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def warps_per_block(kernel: Kernel) -> int:
    """TODO 1: how many warps are needed for a block?"""
    raise NotImplementedError("Fill TODO 1")


def blocks_limited_by_threads(sm: SM, kernel: Kernel) -> int:
    """TODO 2: if thread slots were the only limiter, how many blocks fit on one SM?"""
    raise NotImplementedError("Fill TODO 2")


def blocks_limited_by_registers(sm: SM, kernel: Kernel) -> int:
    regs_per_block = kernel.threads_per_block * kernel.registers_per_thread
    if regs_per_block == 0:
        return sm.max_blocks
    """TODO 3: if registers were the only limiter, how many blocks fit on one SM?"""
    raise NotImplementedError("Fill TODO 3")


def blocks_limited_by_shared_memory(sm: SM, kernel: Kernel) -> int:
    if kernel.shared_mem_per_block == 0:
        return sm.max_blocks
    """TODO 4: if shared memory were the only limiter, how many blocks fit on one SM?"""
    raise NotImplementedError("Fill TODO 4")


def active_blocks_per_sm(sm: SM, kernel: Kernel) -> int:
    return min(
        sm.max_blocks,
        blocks_limited_by_threads(sm, kernel),
        blocks_limited_by_registers(sm, kernel),
        blocks_limited_by_shared_memory(sm, kernel),
    )


def active_warps_per_sm(sm: SM, kernel: Kernel) -> int:
    return active_blocks_per_sm(sm, kernel) * warps_per_block(kernel)


def occupancy(sm: SM, kernel: Kernel) -> float:
    return active_warps_per_sm(sm, kernel) / sm.max_warps


def print_report(sm: SM, kernel: Kernel) -> None:
    blocks = active_blocks_per_sm(sm, kernel)
    warps = active_warps_per_sm(sm, kernel)
    occ = occupancy(sm, kernel)
    print(f"{kernel.name:14s} blocks/SM={blocks:2d} warps/SM={warps:3d} occupancy={occ:5.1%}")


def main() -> None:
    sm = SM()
    kernels = [
        Kernel("matmul_like", threads_per_block=256, registers_per_thread=64, shared_mem_per_block=16 * 1024),
        Kernel("layernorm_like", threads_per_block=256, registers_per_thread=32, shared_mem_per_block=8 * 1024),
        Kernel("relu_like", threads_per_block=128, registers_per_thread=16, shared_mem_per_block=0),
    ]

    print("== Toy occupancy report ==")
    for k in kernels:
        print_report(sm, k)

    print()
    print("思考题：")
    print("1) 哪个 kernel 更可能先被寄存器限制？")
    print("2) 哪个 kernel 更可能 occupancy 高，但仍然未必最快？")
    print("3) 如果把 matmul_like 的 threads_per_block 从 256 改成 512，会发生什么？")


if __name__ == "__main__":
    main()
