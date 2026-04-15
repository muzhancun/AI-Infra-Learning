# Practice scripts

Recommended order:

1. `timeline_gaps.py`
   - Scenario: compare one eager timeline with a more fused timeline and a
     CUDA-graph-like replay.
   - Focus: make launch/host gaps visible, instead of only looking at kernel
     math.
   - Answer: `timeline_gaps_answer.py`

2. `compile_cache_toy.py`
   - Scenario: variable sequence lengths arrive over time, and you want to
     see how exact-shape specialization vs a more dynamic policy changes
     compile-cache behavior.
   - Focus: understand why shape jitter can turn into recompiles.
   - Answer: `compile_cache_toy_answer.py`

These scripts are intentionally simplified. Their job is not to reproduce
PyTorch internals exactly; their job is to train the chapter habit:

> 先看 timeline 的缝，再问这些缝是 launch 造成的，还是 shape 抖动和重编译造成的。
