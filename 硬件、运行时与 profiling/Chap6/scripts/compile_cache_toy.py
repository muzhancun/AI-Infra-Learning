"""
Chapter 6 exercise 2: shape jitter, compile cache, and why "dynamic" helps.

Scenario
--------
Imagine you are serving or training with variable sequence lengths.
Requests arrive in this order:

    (batch, seqlen) = ...

If a compiler specializes on exact shapes, then every new shape may need
a separate compiled entry. If we use a more dynamic strategy, nearby
sequence lengths may be allowed to share one compiled version.

This toy script compares two policies:

1. exact key: cache key is the exact shape
2. bucketed key: batch stays exact, but sequence length is rounded up
   to a coarser bucket

The goal is not to reproduce `torch.compile` exactly. The goal is to make
one runtime idea concrete:

    "Shape jitter can turn into compile cache misses, and a more dynamic
     strategy can trade some specialization for fewer recompiles."

Run
---
python "AI Infra/硬件、运行时与 profiling/Chap6/scripts/compile_cache_toy.py"
"""

# Think of these as consecutive requests observed by one runtime.
# The first coordinate is batch size; the second is sequence length.
SHAPES = [
    (8, 1024),
    (8, 1088),
    (8, 1152),
    (8, 1216),
    (8, 1280),
    (8, 4096),
    (8, 4224),
    (8, 4096),
]

def round_up(x: int, bucket: int) -> int:
    """TODO 1: round x up to the nearest multiple of bucket."""
    raise NotImplementedError("Fill TODO 1")

def exact_key(shape):
    """TODO 2: exact mode uses the shape itself as cache key."""
    raise NotImplementedError("Fill TODO 2")

def bucketed_key(shape, bucket: int = 512):
    """TODO 3: keep batch exact, but bucket sequence length."""
    raise NotImplementedError("Fill TODO 3")

def compile_trace(shapes, key_fn):
    """
    Return a per-request trace of cache behavior.

    Each row answers:
    - what shape arrived?
    - what compile-cache key did we map it to?
    - was that a cache hit, or did we need a new compile?
    """
    seen = set()
    rows = []
    for step, shape in enumerate(shapes, start=1):
        key = key_fn(shape)
        hit = key in seen
        if not hit:
            seen.add(key)
        rows.append((step, shape, key, "hit" if hit else "compile"))
    return rows

def count_compiles(shapes, key_fn) -> int:
    """TODO 4: count how many unique compile cache entries appear."""
    raise NotImplementedError("Fill TODO 4")

def print_trace(title: str, rows) -> None:
    print(title)
    for step, shape, key, status in rows:
        print(f"  step {step:2d}: shape={shape} -> key={key} -> {status}")
    print()

def main() -> None:
    print("Scenario: variable sequence lengths arrive over time.")
    print("Compare exact-shape specialization with a simple bucketed policy.")
    print()
    print("incoming shapes:", SHAPES)
    print()

    exact_rows = compile_trace(SHAPES, exact_key)
    bucket_rows = compile_trace(SHAPES, bucketed_key)
    print_trace("Exact-key trace:", exact_rows)
    print_trace("Bucketed-key trace:", bucket_rows)

    print("exact compiles   :", count_compiles(SHAPES, exact_key))
    print("bucketed compiles:", count_compiles(SHAPES, bucketed_key))
    print()
    print("思考题：")
    print("1) 为什么 sequence length 的小幅抖动，也可能触发 exact-key 模式下的新编译？")
    print("2) 为什么 bucketed / 更动态的策略能减少重编译，但也可能牺牲一部分优化空间？")
    print("3) 这个 bucketed toy 和真实 torch.compile dynamic shapes 有什么差别？")

if __name__ == "__main__":
    main()
