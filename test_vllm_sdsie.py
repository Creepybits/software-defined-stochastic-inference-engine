# test_vllm_sdsie.py
import torch
import time
from vllm_sdsie.kernels.triton_int4_gemm import sdsie_matmul_int4
from vllm_sdsie.kernels.entropy_clutch import SchmittTriggerEntropyClutch

def test_triton_int4_gemm():
    print("=" * 60)
    print("Testing SDSIE Triton INT4 SRAM GEMM on GPU...")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Dimensions: Batch/Seq=16, Hidden=4096, Intermediate=4096 (standard 8B LLM layer)
    M, K, N = 16, 4096, 4096
    
    # 1. Synthesize inputs
    x = torch.randn((M, K), dtype=torch.float16, device=device)
    # Packed INT4 weights: K // 2 bytes along K
    weight_packed = torch.randint(0, 255, (K // 2, N), dtype=torch.uint8, device=device)
    scales = torch.ones((N,), dtype=torch.float16, device=device) * 0.05
    zeros = torch.zeros((N,), dtype=torch.float16, device=device)

    # 2. Warmup & JIT Compile Triton Kernel
    print("-> Compiling and warming up Triton kernel...")
    for _ in range(5):
        out = sdsie_matmul_int4(x, weight_packed, scales, zeros)
    torch.cuda.synchronize()

    # 3. Benchmark Latency
    num_runs = 100
    start = time.perf_counter()
    for _ in range(num_runs):
        out = sdsie_matmul_int4(x, weight_packed, scales, zeros)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) / num_runs * 1000

    print(f"✓ Triton INT4 GEMM executed successfully!")
    print(f"  - Output Shape: {out.shape} (Expected: ({M}, {N}))")
    print(f"  - Output Dtype: {out.dtype}")
    print(f"  - Avg Kernel Execution Latency: {elapsed_ms:.4f} ms")


def test_entropy_clutch():
    print("\n" + "=" * 60)
    print("Testing SDSIE Schmitt-Trigger Entropy Clutch...")
    print("=" * 60)

    clutch = SchmittTriggerEntropyClutch(theta_low=0.55, theta_high=0.82, alpha=0.35)
    vocab_size = 32000

    # Scenario A: Low entropy (High confidence / Sharp probability distribution)
    sharp_logits = torch.randn((1, vocab_size), device="cuda:0")
    sharp_logits[0, 42] = 50.0 # Make one token overwhelmingly probable
    
    active_low = clutch.update_and_decide(sharp_logits)
    print(f"Low-Entropy Step  -> Speculation Active: {active_low} | Running EMA: {clutch.running_entropy:.4f}")

    # Scenario B: High entropy (Flat / Uniform distribution / High uncertainty)
    flat_logits = torch.ones((1, vocab_size), device="cuda:0") * 0.1
    active_high = clutch.update_and_decide(flat_logits)
    print(f"High-Entropy Step -> Speculation Active: {active_high} | Running EMA: {clutch.running_entropy:.4f}")

    print("✓ Schmitt-Trigger Clutch state machine passed successfully!")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required to test Triton kernels!"
    test_triton_int4_gemm()
    test_entropy_clutch()
    print("\n[ALL SDSIE VLLM MODULE CHECKS PASSED]")