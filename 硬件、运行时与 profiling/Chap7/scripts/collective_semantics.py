"""
Chapter 7 exercise 1: collective semantics without GPUs.

Scenario
--------
Imagine 4 ranks participating in distributed training.
Each rank starts with its own local tensor shard or local partial result.
Before worrying about NCCL or topology, you first want to answer:

    "After this collective finishes, who owns the full result,
     and who only owns a shard?"

This toy script uses plain Python lists so that the ownership changes are
visible without any GPU setup.

Run
---
python "AI Infra/硬件、运行时与 profiling/Chap7/scripts/collective_semantics.py"
"""

from typing import List

RankTensors = List[List[int]]
Chunks = List[List[List[int]]]


def all_reduce_sum(ranks: RankTensors) -> RankTensors:
    # Every rank contributes one local tensor.
    # After all-reduce, every rank should receive the same reduced full tensor.
    """TODO 1: every rank gets the elementwise sum of all ranks."""
    raise NotImplementedError("Fill TODO 1")


def all_gather(ranks: RankTensors) -> RankTensors:
    # Each rank starts with only its own shard.
    # After all-gather, every rank should hold the concatenation of all shards.
    """TODO 2: every rank gets all per-rank chunks concatenated in rank order."""
    raise NotImplementedError("Fill TODO 2")


def reduce_scatter_sum(ranks: RankTensors) -> RankTensors:
    # This is "reduce first, then shard".
    # Conceptually:
    # 1) reduce all ranks into one full tensor
    # 2) cut that full tensor evenly back into per-rank shards
    """TODO 3: first reduce elementwise across ranks, then split evenly by rank."""
    raise NotImplementedError("Fill TODO 3")


def all_to_all(chunks_per_rank: Chunks) -> Chunks:
    # chunks_per_rank[src][dst] means:
    # "from source rank src, this is the chunk destined for rank dst".
    # After all-to-all, each destination rank should have received one chunk
    # from every source rank.
    """TODO 4: rank i sends chunk j to destination rank j."""
    raise NotImplementedError("Fill TODO 4")


def main() -> None:
    print("Scenario: 4 ranks, and we only care about ownership semantics.")
    print()

    ranks = [
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        [100, 200, 300, 400],
        [1000, 2000, 3000, 4000],
    ]

    print("all_reduce_sum:")
    print(all_reduce_sum(ranks))
    print()
    print("all_gather:")
    print(all_gather([[1], [2], [3], [4]]))
    print()
    print("reduce_scatter_sum:")
    print(reduce_scatter_sum(ranks))

    routed = [
        [[0, 0], [0, 1], [0, 2], [0, 3]],
        [[1, 0], [1, 1], [1, 2], [1, 3]],
        [[2, 0], [2, 1], [2, 2], [2, 3]],
        [[3, 0], [3, 1], [3, 2], [3, 3]],
    ]
    print()
    print("all_to_all:")
    print(all_to_all(routed))

    print()
    print("思考题：")
    print("1) 为什么 reduce-scatter + all-gather 可以重新组成 all-reduce？")
    print("2) DDP 更像哪一种？MoE token 路由更像哪一种？")
    print("3) 哪些 collective 会让‘完整张量’停留在每个 rank 上？")


if __name__ == "__main__":
    main()
