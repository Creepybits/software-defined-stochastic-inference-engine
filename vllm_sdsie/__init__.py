# verified/vllm_sdsie/__init__.py

__version__ = "2.0.1"
__author__ = "Zanno Jacklin <business@zanno.se>"

from vllm_sdsie.kernels.entropy_clutch import SchmittTriggerEntropyClutch
from vllm_sdsie.spec_decode.sdsie_speculator import SDSIESpeculativeController

__all__ = [
    "SchmittTriggerEntropyClutch",
    "SDSIESpeculativeController",
]
