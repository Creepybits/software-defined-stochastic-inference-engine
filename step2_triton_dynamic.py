import torch
import triton
import triton.language as tl
import time
import json
import os

# =====================================================================
# 1. TRITON HIGH-GEAR SUB-BYTE (INT4 BLOCK-QUANTIZED) GEMM KERNEL
# =====================================================================
@triton.jit
def sdsie_int4_gemm_kernel(
    # Pointers to Matrices
    a_ptr, b_ptr, scales_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_scales_k, stride_scales_n,
    stride_cm, stride_cn,
    # Block size constants
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Sub-byte INT4 Matrix Multiplication with On-Chip SRAM Dequantization.
    Loads packed 4-bit weights across the memory bus, saving 75% bandwidth,
    and unpacks in registers before feeding Tensor Core ALUs.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Memory Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # b_ptr holds packed INT4 weights (2 values per byte along K)
    b_ptrs = b_ptr + ((offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * stride_bn)
    scales_ptrs = scales_ptr + ((offs_k[:, None] // BLOCK_SIZE_K) * stride_scales_k + offs_bn[None, :] * stride_scales_n)

    # Accumulator in FP32 for numerical stability
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load Activations A (BF16)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Load Packed INT4 Weights B (uint8)
        b_packed = tl.load(b_ptrs, mask=(offs_k[:, None] // 2) < (K // 2) - k * (BLOCK_SIZE_K // 2), other=0)
        scales = tl.load(scales_ptrs)

        # Unpack on-chip: Extract Low 4-bits and High 4-bits
        # Even indices take low nibble, Odd indices take high nibble
        shift = (offs_k[:, None] % 2) * 4
        b_unpacked = ((b_packed >> shift) & 0x0F).to(tl.float32) - 8.0  # Zero-point centered at 8
        b_dequant = b_unpacked * scales

        # Tensor Core Matrix Multiply & Accumulate
        accumulator += tl.dot(a.to(tl.float32), b_dequant)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    c = accumulator.to(tl.bfloat16)

    # Store output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def sdsie_int4_matmul(a: torch.Tensor, b_packed: torch.Tensor, scales: torch.Tensor):
    """Python dispatch wrapper for Triton INT4 GEMM kernel"""
    M, K = a.shape
    _, N = scales.shape  # Unpacked N dimension
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


# =====================================================================
# 2. SDSIE DYNAMIC DUAL-GEAR LINEAR MODULE
# =====================================================================
class SDSIEDynamicLinear(torch.nn.Module):
    """
    Seamless Dual-Precision Linear Layer.
    Dispatches to Sub-Byte Triton Kernel in HIGH_GEAR,
    or Native BF16 Tensor Cores in LOW_GEAR.
    """
    def __init__(self, in_features, out_features, block_size_k=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size_k = block_size_k

        # 1. Low Gear Weights (Full BF16)
        self.weight_bf16 = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda") * 0.02
        )

        # 2. High Gear Weights (Packed INT4: 2 weights per byte along in_features)
        # Random initialization packed into uint8
        raw_int4 = torch.randint(0, 16, (in_features, out_features), dtype=torch.uint8, device="cuda")
        self.b_packed = (raw_int4[0::2, :] | (raw_int4[1::2, :] << 4)).contiguous()
        
        # Scales per block
        num_blocks_k = in_features // block_size_k
        self.scales = torch.full((num_blocks_k, out_features), 0.005, dtype=torch.float32, device="cuda")

    def forward(self, x: torch.Tensor, gear: str = "HIGH_GEAR") -> torch.Tensor:
        batch_seq = x.shape[:-1]
        x_2d = x.view(-1, self.in_features)

        if gear == "HIGH_GEAR":
            # Execute sub-byte Triton kernel (75% memory bandwidth reduction)
            out_2d = sdsie_int4_matmul(x_2d, self.b_packed, self.scales)
        else:
            # Fallback to high-precision native BF16 Tensor Core GEMM
            out_2d = torch.matmul(x_2d, self.weight_bf16.t())

        return out_2d.view(*batch_seq, self.out_features)


# =====================================================================
# 3. KERNEL BENCHMARK & MEMORY BANDWIDTH PROFILER
# =====================================================================
def benchmark_triton_dynamic_kernel():
    print("=" * 85)
    print("🚀 PROFILING SDSIE TRITON DYNAMIC QUANTIZATION KERNEL (RTX 5090)")
    print("=" * 85)

    # Test dimensions matching Llama-3.1-8B Feed-Forward (MLP) Projection
    # M=1 (Autoregressive Single Token), K=4096 (Hidden Size), N=14336 (Intermediate Size)
    M = 1
    K = 4096
    N = 14336

    print(f"\nMatrix Dimensions (Llama-3.1 8B MLP Layer):")
    print(f" • Batch/Tokens (M) : {M}")
    print(f" • Input Dim    (K) : {K}")
    print(f" • Output Dim   (N) : {N}\n")

    layer = SDSIEDynamicLinear(in_features=K, out_features=N).cuda()
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    # Warmup both kernel paths
    print("Warming up GPU JIT caches...")
    for _ in range(50):
        _ = layer(x, gear="LOW_GEAR")
        _ = layer(x, gear="HIGH_GEAR")
    torch.cuda.synchronize()

    # Benchmark Low Gear (BF16)
    num_runs = 500
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = layer(x, gear="LOW_GEAR")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    lat_low_gear_us = ((t1 - t0) / num_runs) * 1e6

    # Benchmark High Gear (Triton INT4 Sub-Byte)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = layer(x, gear="HIGH_GEAR")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    lat_high_gear_us = ((t1 - t0) / num_runs) * 1e6

    # Calculate Memory Footprint & Bandwidth
    bytes_bf16 = (K * N * 2)  # 2 bytes per weight
    bytes_int4 = (K * N * 0.5) + (layer.scales.numel() * 4)  # 0.5 bytes per weight + scales

    bw_low_gear = (bytes_bf16 / (lat_low_gear_us * 1e-6)) / 1e9  # GB/s
    bw_high_gear = (bytes_int4 / (lat_high_gear_us * 1e-6)) / 1e9  # GB/s

    mem_reduction = ((bytes_bf16 - bytes_int4) / bytes_bf16) * 100.0
    speedup = lat_low_gear_us / lat_high_gear_us

    print("-" * 85)
    print(f"{'Metric':<32} | {'LOW_GEAR (BF16)':<22} | {'HIGH_GEAR (INT4 Triton)':<22}")
    print("-" * 85)
    print(f"{'Weight Memory Loaded':<32} | {bytes_bf16 / 1e6:<18.2f} MB | {bytes_int4 / 1e6:<18.2f} MB")
    print(f"{'Memory Bus Reduction':<32} | {'Baseline (0%)':<22} | {mem_reduction:<18.1f} %")
    print(f"{'Kernel Latency':<32} | {lat_low_gear_us:<18.2f} µs | {lat_high_gear_us:<18.2f} µs")
    print(f"{'Effective Bus Throughput':<32} | {bw_low_gear:<18.2f} GB/s | {bw_high_gear:<18.2f} GB/s")
    print(f"{'Raw Execution Speedup':<32} | {'1.00x':<22} | {speedup:<18.2f} x")
    print("=" * 85)

    # Save to benchmark ledger
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 5090 32GB",
        "dimensions": {"M": M, "K": K, "N": N},
        "low_gear_bf16": {
            "latency_us": round(lat_low_gear_us, 2),
            "memory_loaded_mb": round(bytes_bf16 / 1e6, 2),
            "bandwidth_gb_s": round(bw_low_gear, 2)
        },
        "high_gear_int4_triton": {
            "latency_us": round(lat_high_gear_us, 2),
            "memory_loaded_mb": round(bytes_int4 / 1e6, 2),
            "bandwidth_gb_s": round(bw_high_gear, 2),
            "memory_reduction_pct": round(mem_reduction, 1),
            "speedup": round(speedup, 2)
        }
    }
    
    os.makedirs(os.path.expanduser("~/sdsie/benchmarks"), exist_ok=True)
    report_file = os.path.expanduser("~/sdsie/benchmarks/triton_kernel_benchmark.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📁 Benchmark report saved to: {report_file}\n")

if __name__ == "__main__":
    benchmark_triton_dynamic_kernel()
