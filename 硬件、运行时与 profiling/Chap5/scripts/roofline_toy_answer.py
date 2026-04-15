"""
Answer for roofline_toy.py.

The roofline here is intentionally tiny and stylized. It is only meant to
connect arithmetic intensity with two ceilings:
1) peak compute
2) peak memory bandwidth
"""
OPS = {"matmul_like": 120.0, "layernorm_like": 1.25, "relu_like": 0.25, "embedding_like": 0.02}
PEAK_TFLOPS = 100.0
MEM_BW_TBPS = 2.0

def memory_limited_tflops(arithmetic_intensity: float, mem_bw_tbps: float) -> float:
    # AI * bandwidth gives the memory-side throughput ceiling.
    return arithmetic_intensity * mem_bw_tbps

def roofline_tflops(arithmetic_intensity: float, peak_tflops: float, mem_bw_tbps: float) -> float:
    # The lower of the compute ceiling and memory ceiling is the roofline.
    return min(peak_tflops, memory_limited_tflops(arithmetic_intensity, mem_bw_tbps))

def classify_bound(arithmetic_intensity: float, peak_tflops: float, mem_bw_tbps: float) -> str:
    # Compare the two ceilings to see which bottleneck is tighter.
    return "compute-bound" if peak_tflops <= memory_limited_tflops(arithmetic_intensity, mem_bw_tbps) else "memory-bound"

def main() -> None:
    print("Scenario: classify which wall each op hits first.")
    print()
    for name, ai in OPS.items():
        print(name, roofline_tflops(ai, PEAK_TFLOPS, MEM_BW_TBPS), classify_bound(ai, PEAK_TFLOPS, MEM_BW_TBPS))

if __name__ == "__main__":
    main()
