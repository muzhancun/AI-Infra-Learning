"""
Answer for collective_semantics.py.

The point of this answer is not the list manipulation itself.
The point is to make the ownership change visible.
"""
from typing import List
RankTensors = List[List[int]]
Chunks = List[List[List[int]]]

def all_reduce_sum(ranks: RankTensors) -> RankTensors:
    reduced = [sum(values) for values in zip(*ranks)]
    return [reduced[:] for _ in ranks]

def all_gather(ranks: RankTensors) -> RankTensors:
    gathered = []
    for chunk in ranks:
        gathered.extend(chunk)
    return [gathered[:] for _ in ranks]

def reduce_scatter_sum(ranks: RankTensors) -> RankTensors:
    reduced = [sum(values) for values in zip(*ranks)]
    nranks = len(ranks)
    chunk = len(reduced) // nranks
    return [reduced[i * chunk : (i + 1) * chunk] for i in range(nranks)]

def all_to_all(chunks_per_rank: Chunks) -> Chunks:
    nranks = len(chunks_per_rank)
    result = [[None for _ in range(nranks)] for _ in range(nranks)]
    for src in range(nranks):
        for dst in range(nranks):
            result[dst][src] = chunks_per_rank[src][dst]
    return result

def main() -> None:
    print("Scenario: inspect who owns full results and who owns shards.")
    print()
    ranks = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400], [1000, 2000, 3000, 4000]]
    print("all_reduce_sum:")
    print(all_reduce_sum(ranks))
    print()
    print("all_gather:")
    print(all_gather([[1], [2], [3], [4]]))
    print()
    print("reduce_scatter_sum:")
    print(reduce_scatter_sum(ranks))
    print()
    routed = [
        [[0, 0], [0, 1], [0, 2], [0, 3]],
        [[1, 0], [1, 1], [1, 2], [1, 3]],
        [[2, 0], [2, 1], [2, 2], [2, 3]],
        [[3, 0], [3, 1], [3, 2], [3, 3]],
    ]
    print("all_to_all:")
    print(all_to_all(routed))

if __name__ == "__main__":
    main()
