# vllm_sdsie/spec_decode/sdsie_speculator.py
import torch
from typing import Optional, Tuple, List
from vllm_sdsie.kernels.entropy_clutch import SchmittTriggerEntropyClutch

class SDSIESpeculativeController:
    """
    Entropy-gated Speculative Decoding Controller for vLLM/SGLang workers.
    Dynamically adjusts draft proposal length (k) or bypasses draft compute entirely
    based on the Schmitt-trigger hysteresis state.
    """
    def __init__(
        self,
        default_k: int = 5,
        theta_low: float = 0.55,
        theta_high: float = 0.82,
        alpha: float = 0.35
    ):
        self.default_k = default_k
        self.clutch = SchmittTriggerEntropyClutch(
            theta_low=theta_low,
            theta_high=theta_high,
            alpha=alpha
        )
        self.total_tokens_generated = 0
        self.speculative_tokens_accepted = 0
        self.draft_bypassed_steps = 0

    def plan_speculation_step(self, current_logits: torch.Tensor) -> int:
        """
        Determines the number of speculative draft tokens to request (k).
        Returns:
            k (int): Number of draft tokens to propose (0 if clutch disengages).
        """
        speculation_active = self.clutch.update_and_decide(current_logits)
        
        if speculation_active:
            return self.default_k
        else:
            self.draft_bypassed_steps += 1
            return 0  # Fallback to pure single-step decoding (saves draft FLOPs)

    def record_verification(self, num_drafted: int, num_accepted: int):
        """
        Telemetry logging for acceptance ratios and energy accounting.
        """
        self.total_tokens_generated += (num_accepted + 1)
        self.speculative_tokens_accepted += num_accepted

    @property
    def acceptance_rate(self) -> float:
        if self.total_tokens_generated == 0:
            return 0.0
        return self.speculative_tokens_accepted / self.total_tokens_generated