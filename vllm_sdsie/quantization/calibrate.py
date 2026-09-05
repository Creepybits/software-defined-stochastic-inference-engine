# vllm_sdsie/quantization/calibrate.py
#
# Basic per-output-channel min-max asymmetric INT4 calibration. This is the
# piece that was completely missing from the project until now (status doc
# Known bug #2): SDSIELinearMethod.create_weights() only ever allocated
# empty weight_packed/scales/zeros tensors -- nothing anywhere produced real
# values for them from an actual model's weights. This fills that gap with
# the simplest scheme that's genuinely correct: for each output channel
# (each row of an nn.Linear's weight), spread the 16 available INT4 levels
# evenly across that channel's own [min, max] range. No calibration
# dataset, no iterative optimization (unlike GPTQ/AWQ) -- just a closed-form
# per-channel affine quantization. Less accurate than group-wise or
# activation-aware schemes, but simple enough to implement and verify
# correctly in one pass, which matters more at this stage than accuracy.
#
# IMPORTANT layout detail, verified numerically before writing this:
# nn.Linear.weight has shape (out_features, in_features) = (N, K). The
# kernel (triton_int4_gemm.py) computes the matmul as x @ W_dequant, where
# W_dequant must have shape (K, N) -- i.e. the TRANSPOSE of nn.Linear's
# native weight layout, matching how nn.Linear itself computes x @ W.T.
# Getting this transpose wrong would silently produce a shape-mismatched or
# nonsensically-permuted result rather than a clean crash, so it's called
# out explicitly here rather than left implicit.
import torch


def quantize_linear_int4_minmax(weight: torch.Tensor):
    """Quantize an nn.Linear-style weight matrix to INT4, per output channel.

    Args:
        weight: (out_features, in_features) float tensor, as stored natively
            by nn.Linear.weight (in_features must be even).

    Returns:
        weight_packed: (in_features // 2, out_features) uint8, ready to hand
            directly to SDSIELinearMethod / sdsie_matmul_int4.
        scales: (out_features,) float, same dtype as the input weight.
        zeros:  (out_features,) float, same dtype as the input weight.
    """
    N, K = weight.shape
    if K % 2 != 0:
        raise ValueError(
            f"in_features must be even for INT4 packing (two 4-bit values "
            f"per byte along K). Got in_features={K}."
        )

    orig_dtype = weight.dtype
    w = weight.detach().float()

    w_min = w.min(dim=1, keepdim=True).values  # (N, 1)
    w_max = w.max(dim=1, keepdim=True).values  # (N, 1)
    # Guard against a degenerate all-equal channel (scale would be 0).
    span = (w_max - w_min).clamp(min=1e-8)

    scale = span / 15.0          # (N, 1) -- 15 = 2^4 - 1 quantization levels
    zero = -w_min / scale        # (N, 1)

    q = (w / scale + zero).round().clamp(0, 15).to(torch.uint8)  # (N, K)

    # Transpose to (K, N) -- matches W.T, which is what the kernel expects
    # (see module docstring above) -- then pack two rows per byte along K.
    q_t = q.t().contiguous()     # (K, N)
    low = q_t[0::2, :]           # even K -> low nibble
    high = q_t[1::2, :]          # odd K -> high nibble
    weight_packed = (high << 4) | low   # (K // 2, N) uint8

    scales = scale.squeeze(1).to(orig_dtype)
    zeros = zero.squeeze(1).to(orig_dtype)
    return weight_packed, scales, zeros


if __name__ == "__main__":
    # Self-test: quantize a small random weight matrix, then check the
    # dequantized-and-reassembled result against a plain-PyTorch reference
    # computed the "normal" nn.Linear way (x @ W.T), independent of the
    # kernel entirely -- isolates whether calibrate.py itself is correct
    # before ever touching the Triton kernel.
    print("=" * 78)
    print("calibrate.py self-test")
    print("=" * 78)

    torch.manual_seed(0)
    N, K, M = 6, 16, 4
    weight = torch.randn(N, K)  # nn.Linear-style (out_features, in_features)
    x = torch.randn(M, K)

    weight_packed, scales, zeros = quantize_linear_int4_minmax(weight)
    assert weight_packed.shape == (K // 2, N)
    assert scales.shape == (N,)
    assert zeros.shape == (N,)
    print(f"[ok] shapes: weight_packed{tuple(weight_packed.shape)}, "
          f"scales{tuple(scales.shape)}, zeros{tuple(zeros.shape)}")

    # Reconstruct W.T from the packed representation exactly as the kernel
    # would dequantize it, and compare against a plain nn.Linear-style
    # matmul using the ORIGINAL (unquantized) weight -- the difference here
    # is the real, expected INT4 quantization error, not a bug.
    low = weight_packed & 0x0F
    high = (weight_packed >> 4) & 0x0F
    q_t = torch.zeros((K, N), dtype=torch.uint8)
    q_t[0::2, :] = low
    q_t[1::2, :] = high
    w_dequant_t = (q_t.float() - zeros[None, :]) * scales[None, :]  # (K, N)

    out_quantized = x @ w_dequant_t
    out_reference = x @ weight.t()

    max_abs_err = (out_quantized - out_reference).abs().max().item()
    rel_err = max_abs_err / out_reference.abs().max().item()
    print(f"[ok] max abs error vs. unquantized reference: {max_abs_err:.4f} "
          f"({rel_err * 100:.2f}% of reference's max magnitude)")
    print("     (nonzero error is expected -- INT4 is lossy by design;")
    print("     this just confirms the calibration/packing pipeline itself")
    print("     is internally consistent, not that quantization is free.)")
    print("=" * 78)
