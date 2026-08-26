# vllm_sdsie/__init__.py
"""
SDSIE: Software-Defined Stochastic Inference Engine
High-efficiency, sub-byte fused SRAM dequantization and entropy-gated speculative decoding.
"""

__version__ = "0.1.0"
__author__ = "Zanno Jacklin <zanno@creepybits.se>"

from vllm_sdsie.kernels.triton_int4_gemm import sdsie_matmul_int4
from vllm_sdsie.kernels.entropy_clutch import SchmittTriggerEntropyClutch
from vllm_sdsie.quantization.sdsie_linear import SDSIELinearMethod
from vllm_sdsie.spec_decode.sdsie_speculator import SDSIESpeculativeController
from vllm_sdsie.patch import patch_vllm

__all__ = [
    "sdsie_matmul_int4",
    "SchmittTriggerEntropyClutch",
    "SDSIELinearMethod",
    "SDSIESpeculativeController",
    "patch_vllm",
]