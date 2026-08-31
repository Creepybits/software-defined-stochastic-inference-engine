import torch
import triton
import triton.language as tl
import time
import json
import os


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "tools", "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)

@triton.jit
def sdsie_int4_gemm_kernel(
    a_ptr, b_ptr, scales_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_scales_k, stride_scales_n,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * stride_bn)
    scales_ptrs = scales_ptr + ((offs_k[:, None] // BLOCK_SIZE_K) * stride_scales_k + offs_bn[None, :] * stride_scales_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_packed = tl.load(b_ptrs, mask=(offs_k[:, None] // 2) < (K // 2) - k * (BLOCK_SIZE_K // 2), other=0)
        scales = tl.load(scales_ptrs)

        shift = (offs_k[:, None] % 2) * 4
        b_unpacked = ((b_packed >> shift) & 0x0F).to(tl.float32) - 8.0
        b_dequant = (b_unpacked * scales).to(tl.bfloat16)

        accumulator += tl.dot(a, b_dequant)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def sdsie_int4_matmul(a: torch.Tensor, b_packed: torch.Tensor, scales: torch.Tensor):
    M, K = a.shape
    _, N = scales.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    sdsie_int4_gemm_kernel[grid](
        a, b_packed, scales, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        scales.stride(0), scales.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8
    )
    return c


class SDSIEDynamicLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, block_size_k=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size_k = block_size_k

        self.weight_bf16 = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda") * 0.02
        )

        raw_int4 = torch.randint(0, 16, (in_features, out_features), dtype=torch.uint8, device="cuda")
        self.b_packed = (raw_int4[0::2, :] | (raw_int4[1::2, :] << 4)).contiguous()
        
        num_blocks_k = in_features // block_size_k
        self.scales = torch.full((num_blocks_k, out_features), 0.005, dtype=torch.bfloat16, device="cuda")

    def forward(self, x: torch.Tensor, gear: str = "HIGH_GEAR") -> torch.Tensor:
        batch_seq = x.shape[:-1]
        x_2d = x.view(-1, self.in_features)

        if gear == "HIGH_GEAR":
            out_2d = sdsie_int4_matmul(x_2d, self.b_packed, self.scales)
        else:
            out_2d = torch.matmul(x_2d, self.weight_bf16.t())

        return out_2d.view(*batch_seq, self.out_features)


def benchmark_triton_dynamic_kernel():
    print("=" * 85)
    print("🚀 PROFILING SDSIE TRITON DYNAMIC QUANTIZATION KERNEL (RTX 5090)")
    print("=" * 85)

    M, K, N = 1, 4096, 14336
    layer = SDSIEDynamicLinear(in_features=K, out_features=N).cuda()
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    for _ in range(50):
        _ = layer(x, gear="LOW_GEAR")
        _ = layer(x, gear="HIGH_GEAR")
    torch.cuda.synchronize()

    num_runs = 500
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = layer(x, gear="LOW_GEAR")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    lat_low_gear_us = ((t1 - t0) / num_runs) * 1e6

    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = layer(x, gear="HIGH_GEAR")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    lat_high_gear_us = ((t1 - t0) / num_runs) * 1e6

    bytes_bf16 = (K * N * 2)
    bytes_int4 = (K * N * 0.5) + (layer.scales.numel() * 2)
    bw_low_gear = (bytes_bf16 / (lat_low_gear_us * 1e-6)) / 1e9
    bw_high_gear = (bytes_int4 / (lat_high_gear_us * 1e-6)) / 1e9
    mem_reduction = ((bytes_bf16 - bytes_int4) / bytes_bf16) * 100.0
    speedup = lat_low_gear_us / lat_high_gear_us

    print(f"Memory Reduction : {mem_reduction:.1f}%")
    print(f"Low Gear Latency : {lat_low_gear_us:.2f} µs")
    print(f"High Gear Latency: {lat_high_gear_us:.2f} µs ({speedup:.2f}x speedup)")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 5090 32GB",
        "dimensions": {"M": M, "K": K, "N": N},
        "low_gear_bf16": {"latency_us": round(lat_low_gear_us, 2), "bandwidth_gb_s": round(bw_low_gear, 2)},
        "high_gear_int4_triton": {"latency_us": round(lat_high_gear_us, 2), "bandwidth_gb_s": round(bw_high_gear, 2), "reduction_pct": round(mem_reduction, 1)}
    }
    
    report_file = os.path.join(TELEMETRY_DIR, "triton_kernel_benchmark.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📁 Benchmark report saved to: {report_file}\n")

if __name__ == "__main__":
    benchmark_triton_dynamic_kernel()
