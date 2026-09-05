# vllm_sdsie/quantization/sdsie_linear.py
#
# FIXED (this session): create_weights() computed packed_k = K // 2 without
# ever checking K was even. For an odd K this silently truncated (integer
# division drops the remainder) rather than raising an error, which would
# have produced a weight_packed tensor one column short of what the real
# checkpoint's K actually needs -- a silent correctness bug, not a crash.
# Now raises a clear error instead. self.packed_bits and the unused `List`
# import (dead code, never referenced anywhere in this class) were removed.
import torch
from typing import Optional
from vllm_sdsie.kernels.triton_int4_gemm import sdsie_matmul_int4

class SDSIELinearMethod:
    """
    Drop-in Linear layer method compatible with vLLM's LinearMethodBase.
    """

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs):

        if input_size_per_partition % 2 != 0:
            raise ValueError(
                f"SDSIELinearMethod packs two 4-bit values per byte along the "
                f"K (input) dimension, which requires an even K. Got "
                f"input_size_per_partition={input_size_per_partition}."
            )

        # Allocate packed weights in VRAM (K // 2, N)
        packed_k = input_size_per_partition // 2
        layer.register_parameter(
            "weight_packed",
            torch.nn.Parameter(torch.empty((packed_k, output_size_per_partition), dtype=torch.uint8), requires_grad=False)
        )
        layer.register_parameter(
            "scales",
            torch.nn.Parameter(torch.empty((output_size_per_partition,), dtype=params_dtype), requires_grad=False)
        )
        layer.register_parameter(
            "zeros",
            torch.nn.Parameter(torch.empty((output_size_per_partition,), dtype=params_dtype), requires_grad=False)
        )

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Executes fused SRAM dequantized GEMM.
        """
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        
        out = sdsie_matmul_int4(x_2d, layer.weight_packed, layer.scales, layer.zeros)
        
        if bias is not None:
            out += bias
            
        return out.view(*orig_shape[:-1], out.shape[-1])


if __name__ == "__main__":
    # End-to-end self-test: create_weights() -> a known packed weight matrix
    # -> apply() -> compare against a plain-PyTorch reference dequant+matmul.
    # This is the first time this file and triton_int4_gemm.py have been
    # checked working *together*, not just individually reviewed -- the
    # kernel needs a real CUDA GPU to run (Triton has no reliable CPU
    # backend for this), so the create_weights() validation check runs
    # unconditionally, and the actual GEMM comparison only runs if CUDA is
    # available.
    print("=" * 78)
    print("SDSIELinearMethod end-to-end self-test")
    print("=" * 78)

    method = SDSIELinearMethod()

    # 1. create_weights() should reject an odd K outright, not silently
    #    truncate it (the bug fixed above).
    dummy_layer = torch.nn.Module()
    try:
        method.create_weights(dummy_layer, input_size_per_partition=7,
                               output_size_per_partition=4, params_dtype=torch.bfloat16)
        print("[FAIL] create_weights() accepted an odd K -- should have raised ValueError.")
    except ValueError as e:
        print(f"[ok] create_weights() correctly rejected odd K: {e}")

    # 2. create_weights() with a valid, even K should allocate the expected
    #    shapes.
    K, N = 8, 4
    method.create_weights(dummy_layer, input_size_per_partition=K,
                           output_size_per_partition=N, params_dtype=torch.bfloat16)
    assert dummy_layer.weight_packed.shape == (K // 2, N)
    assert dummy_layer.scales.shape == (N,)
    assert dummy_layer.zeros.shape == (N,)
    print(f"[ok] create_weights() allocated weight_packed{tuple(dummy_layer.weight_packed.shape)}, "
          f"scales{tuple(dummy_layer.scales.shape)}, zeros{tuple(dummy_layer.zeros.shape)}.")

    if not torch.cuda.is_available():
        print("[skip] No CUDA GPU available -- can't exercise the actual Triton")
        print("       kernel here, only the pure-Python create_weights() checks above.")
    else:
        # 3. Build a small, fully-known example: real (unquantized) integer
        #    nibble values, pack them by hand using the documented convention
        #    (low nibble = even K, high nibble = odd K), and confirm apply()
        #    matches a plain-PyTorch reference computed from the same
        #    unpacked values -- i.e. that packing, dequantization, and the
        #    GEMM genuinely agree with each other end-to-end, not just that
        #    each piece looks right in isolation.
        torch.manual_seed(0)
        M = 3
        device = "cuda"
        dtype = torch.bfloat16

        w_int = torch.randint(0, 16, (K, N), dtype=torch.uint8, device=device)
        scales = (torch.rand(N, device=device) * 0.1 + 0.01).to(dtype)
        zeros = torch.randint(0, 8, (N,), device=device).to(dtype)

        weight_packed = torch.zeros((K // 2, N), dtype=torch.uint8, device=device)
        for j in range(K // 2):
            weight_packed[j] = (w_int[2 * j + 1] << 4) | w_int[2 * j]

        real_layer = torch.nn.Module()
        real_layer.weight_packed = weight_packed
        real_layer.scales = scales
        real_layer.zeros = zeros

        x = torch.randn(M, K, dtype=dtype, device=device)
        bias = torch.randn(N, dtype=dtype, device=device)

        out = method.apply(real_layer, x, bias=bias)

        w_dequant_ref = (w_int.to(dtype) - zeros[None, :]) * scales[None, :]
        expected = x @ w_dequant_ref + bias

        max_abs_diff = (out - expected).abs().max().item()
        # bfloat16 has ~2-3 decimal digits of precision; this tolerance is
        # generous enough to catch a real logic bug while tolerating normal
        # bf16 rounding.
        ok = torch.allclose(out, expected, atol=5e-2, rtol=5e-2)
        print(f"[{'ok' if ok else 'FAIL'}] apply() vs. plain-PyTorch reference: "
              f"max abs diff = {max_abs_diff:.4f}")
        print(f"     output dtype: {out.dtype} (should match input dtype {dtype})")

    print("=" * 78)