"""
Chapter 5 exercise 2: use a toy roofline to continue the diagnosis.

Scenario
--------
Suppose you already formed a rough arithmetic-intensity estimate for a few
ops. Now you want to ask the next question:

    "On this toy GPU, which ops are more likely limited by memory
     bandwidth, and which ones are more likely limited by peak compute?"

That is the role of the roofline intuition in this chapter. It is not a
full performance model. It is a compact way to connect:

- arithmetic intensity from exercise 1
- peak TFLOPS of the machine
- memory bandwidth of the machine

If you finished exercise 1, feel free to replace the default AI values
below with your own outputs.

Run
---
python "AI Infra/硬件、运行时与 profiling/Chap5/scripts/roofline_toy.py"
"""

# Representative AI values for four different "personalities" of ops.
# They are stylized on purpose:
# - matmul_like: high AI, more likely to approach compute limits
# - layernorm_like / relu_like: lower AI, easier to become memory-bound
# - embedding_like: almost pure movement
OPS = {"matmul_like": 120.0, "layernorm_like": 1.25, "relu_like": 0.25, "embedding_like": 0.02}
PEAK_TFLOPS = 100.0
MEM_BW_TBPS = 2.0

def memory_limited_tflops(arithmetic_intensity: float, mem_bw_tbps: float) -> float:
    # FLOPs/Byte * Bytes/s = FLOPs/s.
    # In this toy script we keep the units simplified so the result reads
    # directly as TFLOPS.
    """TODO 1: FLOPs/Byte * Bytes/s -> FLOPs/s, simplified as TFLOPS."""
    raise NotImplementedError("Fill TODO 1")

def roofline_tflops(arithmetic_intensity: float, peak_tflops: float, mem_bw_tbps: float) -> float:
    # Roofline intuition:
    # - compute ceiling says "you cannot go above peak_tflops"
    # - memory ceiling says "with this AI and this bandwidth, you also
    #   cannot go above AI * bandwidth"
    # The lower ceiling wins.
    """TODO 2: roofline says achieved throughput is min(peak, memory_limit)."""
    raise NotImplementedError("Fill TODO 2")

def classify_bound(arithmetic_intensity: float, peak_tflops: float, mem_bw_tbps: float) -> str:
    # If the memory ceiling is below peak compute, the op is memory-bound.
    # Otherwise the compute ceiling is the tighter limit.
    """TODO 3: return memory-bound or compute-bound."""
    raise NotImplementedError("Fill TODO 3")

def main() -> None:
    print("Scenario: you already have rough AI estimates.")
    print("Now put them on a toy GPU roofline to judge which wall each op")
    print("hits first: the bandwidth wall or the compute wall.")
    print()

    for name, ai in OPS.items():
        achieved = roofline_tflops(ai, PEAK_TFLOPS, MEM_BW_TBPS)
        print(f"{name:16s} AI={ai:7.2f}  roofline={achieved:7.2f} TFLOPS  {classify_bound(ai, PEAK_TFLOPS, MEM_BW_TBPS)}")
    print()
    print("思考题：")
    print("1) 如果 trace 里某类 op 的 AI 很低，为什么它更容易先撞上带宽墙，而不是算力墙？")
    print("2) 为什么 matmul_like 更可能受益于更好的数据复用和更高的计算利用率？")
    print("3) 如果带宽翻倍但 peak TFLOPS 不变，哪些 op 最受益？这会怎样反映在 profiler 的时间占比里？")

if __name__ == "__main__":
    main()
