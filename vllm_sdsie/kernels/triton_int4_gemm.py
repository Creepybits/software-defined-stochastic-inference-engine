# vllm_sdsie/kernels/triton_int4_gemm.py
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _sdsie_int4_gemm_kernel(
    # Pointers to Matrices
    a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scales_n,
    stride_zeros_n,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    """
    Sub-byte INT4 Matrix Multiplication with on-the-fly SRAM dequantization.
    Weights (B) are stored packed as int8/uint8 (2 x 4-bit values per byte along K).
    Decompression happens inside registers before fused FMA operations.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k_packed = tl.arange(0, BLOCK_K // 2)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (offs_k_packed[None, :] * 2) * stride_ak)
    # Packed B has shape (K // 2, N)
    b_ptrs = b_ptr + (offs_k_packed[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Load scales and zeros for the current N block
    scales = tl.load(scales_ptr + offs_bn * stride_scales_n)
    zeros = tl.load(zeros_ptr + offs_bn * stride_zeros_n)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load packed INT4 weights (2 values per byte)
        b_packed = tl.load(b_ptrs) # shape: (BLOCK_K // 2, BLOCK_N)
        
        # Sub-byte unpack into low and high 4-bit nibbles
        b_low = (b_packed & 0x0F).to(tl.float16)
        b_high = ((b_packed >> 4) & 0x0F).to(tl.float16)

        # Dequantize in-register: W = (Q - Z) * S
        b_low_dequant = (b_low - zeros[None, :]) * scales[None, :]
        b_high_dequant = (b_high - zeros[None, :]) * scales[None, :]

        # Interleave along K-dimension
        # Load corresponding activation tiles A (FP16/BF16)
        a_tile_0 = tl.load(a_ptrs)
        a_tile_1 = tl.load(a_ptrs + stride_ak)

        # Fused Multiply-Accumulate
        accumulator += tl.dot(a_tile_0, b_low_dequant)
        accumulator += tl.dot(a_tile_1, b_high_dequant)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk

    c = accumulator.to(tl.float16)
    
    # Store output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def sdsie_matmul_int4(a: torch.Tensor, b_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for Triton sub-byte GEMM.
    a: (M, K) float16 / bfloat16
    b_packed: (K // 2, N) uint8 / int8
    scales: (N,) float16
    zeros: (N,) float16
    """
    M, K = a.shape
    K_packed, N = b_packed.shape
    assert K_packed == K // 2, f"Dimension mismatch: A has K={K}, packed B has K_packed={K_packed}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    _sdsie_int4_gemm_kernel[grid](
        a, b_packed, c, scales, zeros,
        M, N, K,
        a.stride(0), a.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        c.stride(0), c.stride(1),
        scales.stride(0),
        zeros.stride(0),
    )
    return c