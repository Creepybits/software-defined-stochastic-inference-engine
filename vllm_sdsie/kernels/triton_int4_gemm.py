# vllm_sdsie/kernels/triton_int4_gemm.py
#
# FIXED (this session): two real correctness gaps, both confirmed by direct
# review and fixed rather than just documented:
#   1. K-dimension loads inside the accumulation loop had no bounds masking
#      -- only the final output store did -- so the kernel read out of
#      bounds past the end of K whenever K wasn't an exact multiple of
#      BLOCK_K. Now masked, so it works for any K (previously only "safe"
#      for Llama-3.1-8B's 4096/14336 dims, which happen to be multiples of
#      the fixed BLOCK_K=64).
#   2. Both the dequantized weights and the final output were hardcoded to
#      tl.float16, regardless of the activation tensor's actual dtype. This
#      project loads every model in bfloat16, and sdsie_linear.py allocates
#      scales/zeros in whatever dtype the model actually uses -- so the
#      hardcoded float16 casts didn't match either the surrounding code or
#      this project's real usage. Now computed in the activation tensor's
#      actual dtype instead.
# Per-output-channel-only quantization granularity (no per-K-group scale)
# is unchanged and still a real accuracy/coarseness limitation, not fixed
# here -- see the status doc.
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

    # Compute in whatever dtype the activations actually are (float16 or
    # bfloat16) rather than hardcoding float16 -- this project loads all its
    # models in bfloat16, and the original hardcoded .to(tl.float16) casts
    # here (and on the final output below) didn't match that, or the
    # bfloat16 scales/zeros tensors sdsie_linear.py actually allocates.
    compute_dtype = a_ptr.dtype.element_ty

    for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
        # Boundary-safe masks for the K dimension. The original version had
        # no masking on any load inside this loop -- only the final output
        # store was masked -- so it silently read out of bounds past the end
        # of K whenever K wasn't an exact multiple of BLOCK_K. M and N don't
        # need masks here: offs_am/offs_bn are already wrapped with modulo
        # above, so those reads always land on valid (if sometimes unused)
        # memory, and get discarded by the real mask at the final store.
        k_packed_idx = k_iter * (BLOCK_K // 2) + offs_k_packed
        k_even_idx = k_iter * BLOCK_K + offs_k_packed * 2
        k_odd_idx = k_even_idx + 1

        b_mask = k_packed_idx[:, None] < (K // 2)
        a_mask_0 = k_even_idx[None, :] < K
        a_mask_1 = k_odd_idx[None, :] < K

        # Load packed INT4 weights (2 values per byte)
        b_packed = tl.load(b_ptrs, mask=b_mask, other=0)  # shape: (BLOCK_K // 2, BLOCK_N)

        # Sub-byte unpack into low and high 4-bit nibbles
        b_low = (b_packed & 0x0F).to(compute_dtype)
        b_high = ((b_packed >> 4) & 0x0F).to(compute_dtype)

        # Dequantize in-register: W = (Q - Z) * S
        b_low_dequant = (b_low - zeros[None, :]) * scales[None, :]
        b_high_dequant = (b_high - zeros[None, :]) * scales[None, :]

        # Interleave along K-dimension
        # Load corresponding activation tiles A (FP16/BF16)
        a_tile_0 = tl.load(a_ptrs, mask=a_mask_0, other=0.0)
        a_tile_1 = tl.load(a_ptrs + stride_ak, mask=a_mask_1, other=0.0)

        # Fused Multiply-Accumulate
        accumulator += tl.dot(a_tile_0, b_low_dequant)
        accumulator += tl.dot(a_tile_1, b_high_dequant)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)
    
    # Store output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def sdsie_matmul_int4(a: torch.Tensor, b_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for Triton sub-byte GEMM.
    a: (M, K) float16 or bfloat16 -- output dtype and internal compute dtype
       both follow whatever dtype `a` actually is (see kernel-level fix note
       at the top of this file).
    b_packed: (K // 2, N) uint8 / int8
    scales: (N,) matches a's dtype
    zeros: (N,) matches a's dtype
    K does not need to be an exact multiple of the kernel's BLOCK_K (masked
    correctly either way); it does need to be even, since two 4-bit values
    are packed per byte along K.
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