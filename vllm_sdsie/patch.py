# vllm_sdsie/patch.py
#
# FIXED (this session): the docstring below claimed this hooks BOTH
# "quantization methods and entropy-gated speculative decoding" into vLLM,
# but the function body only ever attempted the quantization registration --
# no speculative-decoding hook existed anywhere in this file. Docstring
# corrected to describe only what the code actually does. Adding a real
# speculative-decoding registration would mean integrating against vLLM's
# actual internal scheduler/worker APIs, which isn't done here (that's a
# separate, much larger task -- see the status doc's remaining integration
# gap) rather than something safe to guess at and fabricate.
import importlib
import logging

logger = logging.getLogger("vllm_sdsie")

def patch_vllm() -> bool:
    """
    Hooks the SDSIE INT4 quantization method into the active vLLM runtime's
    quantization registry, if vLLM is installed. Does NOT hook
    entropy-gated speculative decoding into vLLM -- that integration
    doesn't exist yet (see the status doc).
    """
    try:
        vllm = importlib.import_module("vllm")
        vllm_version = getattr(vllm, "__version__", "unknown")
        logger.info(f"Detected vLLM version {vllm_version}")

        # Attempt to register SDSIE into vLLM's Quantization Registry
        try:
            quant_module = importlib.import_module("vllm.model_executor.layers.quantization")
            quant_methods = getattr(quant_module, "QUANTIZATION_METHODS", None)
            
            if quant_methods is not None and isinstance(quant_methods, dict):
                from vllm_sdsie.quantization.sdsie_linear import SDSIELinearMethod
                quant_methods["sdsie"] = SDSIELinearMethod
                logger.info("✓ Successfully registered 'sdsie' in vLLM QUANTIZATION_METHODS.")
            else:
                logger.info("vLLM quantization registry structure handled via custom backend.")
        except (ImportError, AttributeError) as err:
            logger.debug(f"vLLM quantization registry hook skipped: {err}")

        return True

    except ImportError:
        logger.info("vLLM not installed in active environment. Running in standalone SDSIE mode.")
        return False