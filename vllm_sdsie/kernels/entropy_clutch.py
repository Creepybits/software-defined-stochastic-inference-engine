# vllm_sdsie/kernels/entropy_clutch.py
import torch
import math

class SchmittTriggerEntropyClutch:
    """
    Industrial Schmitt-Trigger Hysteresis Clutch for Speculative Decoding.
    Eliminates draft-model thrashing by maintaining dual entropy thresholds.
    """
    def __init__(self, theta_low: float = 0.55, theta_high: float = 0.82, alpha: float = 0.35):
        self.theta_low = theta_low     # Threshold to re-engage speculative draft
        self.theta_high = theta_high   # Threshold to disengage draft (fallback)
        self.alpha = alpha             # EMA smoothing coefficient
        self.running_entropy = 0.0
        self.speculation_active = True

    def compute_token_entropy(self, logits: torch.Tensor) -> float:
        """
        Computes Shannon entropy: H(p) = -sum(p * log(p)) on the top logits.
        """
        probs = torch.softmax(logits[-1, :], dim=-1)
        # Numerical stability clamp
        probs = torch.clamp(probs, min=1e-9, max=1.0)
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        return entropy

    def update_and_decide(self, logits: torch.Tensor) -> bool:
        """
        Calculates EMA entropy and updates hysteresis state.
        Returns: True if speculative draft should execute, False for single-step fallback.
        """
        step_entropy = self.compute_token_entropy(logits)
        
        # Initialize EMA if first step
        if self.running_entropy == 0.0:
            self.running_entropy = step_entropy
        else:
            self.running_entropy = self.alpha * step_entropy + (1.0 - self.alpha) * self.running_entropy

        # Schmitt-trigger state machine
        if self.speculation_active and self.running_entropy > self.theta_high:
            self.speculation_active = False  # Disengage clutch (high uncertainty/reasoning fork)
        elif not self.speculation_active and self.running_entropy < self.theta_low:
            self.speculation_active = True   # Re-engage clutch (low uncertainty/predictable text)

        return self.speculation_active