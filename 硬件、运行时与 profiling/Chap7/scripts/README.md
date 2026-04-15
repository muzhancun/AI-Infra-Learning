# Practice scripts

Recommended order:

1. `collective_semantics.py`
   - Scenario: start from 4 ranks on plain Python lists and ask who owns the
     final full tensor, and who only keeps a shard.
   - Focus: understand semantic differences between all-reduce, all-gather,
     reduce-scatter, and all-to-all before worrying about hardware.
   - Answer: `collective_semantics_answer.py`

2. `collective_cost_toy.py`
   - Scenario: once the semantics are clear, compare how many bytes each rank
     roughly needs to move for different collectives.
   - Focus: connect ownership changes with communication cost.
   - Answer: `collective_cost_toy_answer.py`

These scripts are intentionally simplified. Their job is not to reproduce
NCCL internals exactly; their job is to train the chapter habit:

> 先问张量最后归谁，再问为了让它归到那里，每个 rank 要搬多少。
