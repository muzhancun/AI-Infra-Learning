"""
Chapter 7 exercise 2: toy byte accounting for collectives.

Scenario
--------
Suppose one logical full tensor is 1 GiB and 8 ranks participate.
Once you already understand the semantics, the next question is:

    "Roughly how many bytes does each rank need to move?"

This script uses simplified per-rank accounting to compare several
collectives. It is not a full NCCL model; it is a way to connect semantic
choices with rough communication cost.

Run
---
python "AI Infra/硬件、运行时与 profiling/Chap7/scripts/collective_cost_toy.py"
"""
FULL_BYTES = 1024 * 1024 * 1024
NRANKS = 8

def all_gather_bytes_per_rank(full_tensor_bytes: int, nranks: int) -> float:
    # Each rank starts with 1/n shard, so it already owns 1/n of the full
    # tensor. To reconstruct the full tensor locally, it must receive the
    # other (n-1)/n portion.
    """TODO 1: each rank starts with 1/n shard and receives the other shards."""
    raise NotImplementedError("Fill TODO 1")

def reduce_scatter_bytes_per_rank(full_tensor_bytes: int, nranks: int) -> float:
    # Simplified view: ring reduce-scatter moves roughly the same total
    # bytes per rank as all-gather, but the semantic result is different.
    """TODO 2: simplified ring-style accounting."""
    raise NotImplementedError("Fill TODO 2")

def all_reduce_ring_bytes_per_rank(full_tensor_bytes: int, nranks: int) -> float:
    # Ring all-reduce can be viewed as:
    # reduce-scatter + all-gather
    """TODO 3: simplified ring all-reduce is reduce-scatter + all-gather."""
    raise NotImplementedError("Fill TODO 3")

def all_to_all_bytes_per_rank(full_tensor_bytes: int, nranks: int) -> float:
    # Rough view: each rank sends different shards out and receives different
    # shards back, so send+receive are both part of the story.
    """TODO 4: rough send+receive accounting when every rank exchanges different shards."""
    raise NotImplementedError("Fill TODO 4")

def main() -> None:
    print("Scenario: compare rough per-rank communication volume.")
    print(f"full tensor bytes = {FULL_BYTES}")
    print(f"nranks            = {NRANKS}")
    print()
    print("all_gather     :", all_gather_bytes_per_rank(FULL_BYTES, NRANKS))
    print("reduce_scatter :", reduce_scatter_bytes_per_rank(FULL_BYTES, NRANKS))
    print("all_reduce     :", all_reduce_ring_bytes_per_rank(FULL_BYTES, NRANKS))
    print("all_to_all     :", all_to_all_bytes_per_rank(FULL_BYTES, NRANKS))
    print()
    print("思考题：")
    print("1) 为什么 all-reduce 常常比 all-gather 更贵？")
    print("2) 为什么 reduce-scatter + all-gather 是一个重要组合？")
    print("3) 为什么 all-to-all 往往更容易暴露拓扑和负载不均的问题？")

if __name__ == "__main__":
    main()
