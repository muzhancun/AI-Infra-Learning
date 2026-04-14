"""
Chapter 4 exercise 2: toy block scheduler.
Run:
python "AI Infra/硬件、运行时与 profiling/Chap4/scripts/block_scheduler_toy.py"
"""

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def round_robin_assign(num_sms: int, num_blocks: int):
    """TODO 1: assign blocks to SMs in a simple round-robin way."""
    # raise NotImplementedError("Fill TODO 1")
    return 

def idle_sms(assignment):
    """TODO 2: count how many SMs got zero blocks."""
    raise NotImplementedError("Fill TODO 2")

def waves(num_sms: int, blocks_per_sm_limit: int, num_blocks: int) -> int:
    """TODO 3: how many waves are needed if an SM can host only blocks_per_sm_limit blocks at once?"""
    raise NotImplementedError("Fill TODO 3")

def utilization(assignment) -> float:
    """TODO 4: fraction of SMs that got at least one block."""
    raise NotImplementedError("Fill TODO 4")

def main() -> None:
    cases = [(8, 4, 2), (8, 16, 2), (8, 40, 2)]
    for num_sms, num_blocks, blocks_per_sm_limit in cases:
        a = round_robin_assign(num_sms, num_blocks)
        print(f"num_sms={num_sms}, num_blocks={num_blocks}, limit={blocks_per_sm_limit}")
        print(" assignment:", a)
        print(" idle_sms  :", idle_sms(a))
        print(" utilization:", f"{utilization(a):.1%}")
        print(" waves     :", waves(num_sms, blocks_per_sm_limit, num_blocks))
        print()
    print("思考题：")
    print("1) 为什么 block 必须彼此独立，调度器才能放心地把它们分给不同 SM？")
    print("2) 为什么 block 太少时，GPU 可能还没忙起来？")
    print("3) 为什么 occupancy 和 block 总数都只是线索，不是最后的性能答案？")

if __name__ == "__main__":
    main()
