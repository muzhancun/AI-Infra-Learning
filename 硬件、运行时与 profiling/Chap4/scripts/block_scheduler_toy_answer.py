"""Answer for block_scheduler_toy.py."""

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def round_robin_assign(num_sms: int, num_blocks: int):
    assignment = [0] * num_sms
    for block_id in range(num_blocks):
        assignment[block_id % num_sms] += 1
    return assignment

def idle_sms(assignment):
    return sum(1 for x in assignment if x == 0)

def waves(num_sms: int, blocks_per_sm_limit: int, num_blocks: int) -> int:
    return ceil_div(num_blocks, num_sms * blocks_per_sm_limit)

def utilization(assignment) -> float:
    return sum(1 for x in assignment if x > 0) / len(assignment)

def main() -> None:
    cases = [(8, 4, 2), (8, 16, 2), (8, 40, 2)]
    for num_sms, num_blocks, blocks_per_sm_limit in cases:
        a = round_robin_assign(num_sms, num_blocks)
        print(num_sms, num_blocks, blocks_per_sm_limit, a, idle_sms(a), utilization(a), waves(num_sms, blocks_per_sm_limit, num_blocks))

if __name__ == "__main__":
    main()
