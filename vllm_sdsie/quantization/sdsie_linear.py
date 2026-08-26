# vllm_sdsie/quantization/sdsie_linear.py
import torch
from typing import Optional, List
from vllm_sdsie.kernels.triton_int4_gemm import sdsie_matmul_int4

class SDSIELinearMethod:
    """
    Drop-in Linear layer method compatible with vLLM's LinearMethodBase.
    """
    def __init__(self):
        self.packed_bits = 4

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs):
        
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