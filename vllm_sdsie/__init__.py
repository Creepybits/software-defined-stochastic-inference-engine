# verified/vllm_sdsie/__init__.py
"""
SDSIE verified components only.
See VERIFIED_MANIFEST.md for what's been checked and how.
Mirrors the structure of the main repo's vllm_sdsie/, but only
contains files that have passed verification.
"""

__version__ = "2.0.1"
__author__ = "Zanno Jacklin <business@zanno.se>"

from vllm_sdsie.kernels.entropy_clutch import SchmittTriggerEntropyClutch
from vllm_sdsie.spec_decode.sdsie_speculator import SDSIESpeculativeController

__all__ = [
    "SchmittTriggerEntropyClutch",
    "SDSIESpeculativeController",
]
