

__version__ = "2.0.2"
__author__ = "Zanno Jacklin <business@zanno.se>"

from vllm_sdsie.kernels.entropy_clutch import SchmittTriggerEntropyClutch
from vllm_sdsie.spec_decode.sdsie_speculator import SDSIESpeculativeController
from vllm_sdsie.patch import patch_vllm

__all__ = [
    "SchmittTriggerEntropyClutch",
    "SDSIESpeculativeController",
    "patch_vllm",
]

# FIXED (this session): patch_vllm() was defined but never called anywhere
# in the codebase -- importing this package didn't actually register
# SDSIELinearMethod with vLLM even when vLLM was installed, despite that
# being the whole stated purpose of patch.py. Calling it here is safe
# regardless of what's installed: patch_vllm() internally catches the
# ImportError for a missing vLLM, and separately catches ImportError for a
# missing triton (which only gets imported transitively, inside the
# try/except, if vLLM's quantization registry is actually found) -- neither
# missing dependency raises out of this call.
patch_vllm()