"""
Answer for compile_cache_toy.py.

This answer shows one simple idea: exact specialization sees more distinct
cache keys, while a coarser dynamic policy lets nearby shapes share.
"""
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
    return ((x + bucket - 1) // bucket) * bucket

def exact_key(shape):
    return shape

def bucketed_key(shape, bucket: int = 512):
    bsz, seqlen = shape
    return bsz, round_up(seqlen, bucket)

def compile_trace(shapes, key_fn):
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
    seen = set()
    for shape in shapes:
        seen.add(key_fn(shape))
    return len(seen)

def print_trace(title: str, rows) -> None:
    print(title)
    for step, shape, key, status in rows:
        print(f"  step {step:2d}: shape={shape} -> key={key} -> {status}")
    print()

def main() -> None:
    print("Scenario: read compile-cache behavior request by request.")
    print()
    exact_rows = compile_trace(SHAPES, exact_key)
    bucket_rows = compile_trace(SHAPES, bucketed_key)
    print_trace("Exact-key trace:", exact_rows)
    print_trace("Bucketed-key trace:", bucket_rows)
    print("exact compiles   :", count_compiles(SHAPES, exact_key))
    print("bucketed compiles:", count_compiles(SHAPES, bucketed_key))

if __name__ == "__main__":
    main()
