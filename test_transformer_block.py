# test_transformer_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from vllm_sdsie.kernels.triton_int4_gemm import sdsie_matmul_int4

class SDSIEMLP(nn.Module):
    """
    Standard Llama-style SwiGLU MLP block using SDSIE Sub-Byte Dequantization.
    x -> (gate(x) * swish(up(x))) -> down(x)
    """
    def __init__(self, hidden_dim=4096, intermediate_dim=11008, device="cuda:0"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.device = device

        # Packed INT4 weights: (K // 2, N)
        self.gate_packed = nn.Parameter(torch.randint(0, 255, (hidden_dim // 2, intermediate_dim), dtype=torch.uint8, device=device), requires_grad=False)
        self.up_packed   = nn.Parameter(torch.randint(0, 255, (hidden_dim // 2, intermediate_dim), dtype=torch.uint8, device=device), requires_grad=False)
        self.down_packed = nn.Parameter(torch.randint(0, 255, (intermediate_dim // 2, hidden_dim), dtype=torch.uint8, device=device), requires_grad=False)

        # Scales & Zeros
        self.gate_scale = nn.Parameter(torch.ones((intermediate_dim,), dtype=torch.float16, device=device) * 0.02, requires_grad=False)
        self.up_scale   = nn.Parameter(torch.ones((intermediate_dim,), dtype=torch.float16, device=device) * 0.02, requires_grad=False)
        self.down_scale = nn.Parameter(torch.ones((hidden_dim,), dtype=torch.float16, device=device) * 0.02, requires_grad=False)

        self.gate_zero  = nn.Parameter(torch.zeros((intermediate_dim,), dtype=torch.float16, device=device), requires_grad=False)
        self.up_zero    = nn.Parameter(torch.zeros((intermediate_dim,), dtype=torch.float16, device=device), requires_grad=False)
        self.down_zero  = nn.Parameter(torch.zeros((hidden_dim,), dtype=torch.float16, device=device), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Gate projection
        gate = sdsie_matmul_int4(x, self.gate_packed, self.gate_scale, self.gate_zero)
        # 2. Up projection
        up = sdsie_matmul_int4(x, self.up_packed, self.up_scale, self.up_zero)
        # 3. SwiGLU Activation: silu(gate) * up
        activated = F.silu(gate) * up
        # 4. Down projection
        out = sdsie_matmul_int4(activated, self.down_packed, self.down_scale, self.down_zero)
        return out


def benchmark_mlp():
    print("=" * 60)
    print("Benchmarking SDSIE Llama-3-8B SwiGLU MLP Block...")
    print("=" * 60)
    device = torch.device("cuda:0")

    mlp = SDSIEMLP(hidden_dim=4096, intermediate_dim=14336, device=device)
    
    # Token decode batch (e.g. Batch=8 tokens)
    x = torch.randn((8, 4096), dtype=torch.float16, device=device)

    # Warmup
    for _ in range(10):
        _ = mlp(x)
    torch.cuda.synchronize()

    # Benchmark 200 passes
    runs = 200
    start = time.perf_counter()
    for _ in range(runs):
        out = mlp(x)
    torch.cuda.synchronize()
    total_time_ms = (time.perf_counter() - start) / runs * 1000

    print(f"✓ SwiGLU Forward Pass Successful!")
    print(f"  - Input shape:  {x.shape}")
    print(f"  - Output shape: {out.shape}")
    print(f"  - Total MLP Layer Latency: {total_time_ms:.4f} ms")


if __name__ == "__main__":
    benchmark_mlp()