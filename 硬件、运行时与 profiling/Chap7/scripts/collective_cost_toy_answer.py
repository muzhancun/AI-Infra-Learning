"""
Answer for collective_cost_toy.py.

This answer keeps only a rough per-rank byte model. The purpose is to see
why semantic choices often imply different communication costs.
"""
FULL_BYTES = 1024 * 1024 * 1024
NRANKS = 8

def all_gather_bytes_per_rank(full_tensor_bytes: int, nranks: int) -> float:
    return (nranks - 1) / nranks * full_tensor_bytes

def reduce_scatter_bytes_per_rank(full_tensor_bytes: int, nranks: int) -> float:
    return (nranks - 1) / nranks * full_tensor_bytes

def all_reduce_ring_bytes_per_rank(full_tensor_bytes: int, nranks: int) -> float:
    return 2 * (nranks - 1) / nranks * full_tensor_bytes

def all_to_all_bytes_per_rank(full_tensor_bytes: int, nranks: int) -> float:
    return 2 * (nranks - 1) / nranks * full_tensor_bytes

def main() -> None:
    print("Scenario: compare rough per-rank communication volume.")
    print()
    print("all_gather     :", all_gather_bytes_per_rank(FULL_BYTES, NRANKS))
    print("reduce_scatter :", reduce_scatter_bytes_per_rank(FULL_BYTES, NRANKS))
    print("all_reduce     :", all_reduce_ring_bytes_per_rank(FULL_BYTES, NRANKS))
    print("all_to_all     :", all_to_all_bytes_per_rank(FULL_BYTES, NRANKS))

if __name__ == "__main__":
    main()
