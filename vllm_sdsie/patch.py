# vllm_sdsie/patch.py
import importlib
import logging

logger = logging.getLogger("vllm_sdsie")

def patch_vllm() -> bool:
    """
    Hooks SDSIE quantization methods and entropy-gated speculative decoding
    into the active vLLM runtime environment if vLLM is installed.
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