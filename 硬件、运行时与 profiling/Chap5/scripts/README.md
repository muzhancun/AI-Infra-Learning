# Practice scripts

Recommended order:

1. `arithmetic_intensity_toy.py`
   - Scenario: you are looking at a trace and want a first-pass judgment of
     which ops are doing a lot of math per byte moved, and which ones are
     mostly paying for data movement.
   - Focus: build intuition for FLOPs/Byte.
   - Answer: `arithmetic_intensity_toy_answer.py`

2. `roofline_toy.py`
   - Scenario: after getting a rough AI estimate, put each op on a toy
     roofline and ask whether it is more likely limited by memory
     bandwidth or by peak compute.
   - Focus: connect AI with bandwidth-vs-compute diagnosis.
   - Answer: `roofline_toy_answer.py`

These scripts are intentionally simplified. Their job is not exact hardware
prediction; their job is to train the chapter habit:

> 看到一个慢 op，先问它是算太少，还是搬太多。
