# vllm_sdsie/kernels/entropy_clutch.py
import torch

class SchmittTriggerEntropyClutch:
    """
    Industrial Schmitt-Trigger Hysteresis Clutch for Speculative Decoding.
    Numerically stable for large-vocabulary LLMs (128k+ tokens).
    """
    def __init__(self, theta_low: float = 0.55, theta_high: float = 1.25, alpha: float = 0.35):
        self.theta_low = theta_low     # Threshold to re-engage speculative draft
        self.theta_high = theta_high   # Threshold to disengage draft (fallback)
        self.alpha = alpha             # EMA smoothing coefficient
        self.running_entropy = 0.0
        self.speculation_active = True

    def compute_token_entropy(self, logits: torch.Tensor) -> float:
        """
        Computes Shannon entropy: H(p) = -sum(p * log2(p)) in float32.
        """
        # Flatten to 1D and cast to float32 to prevent FP16 overflow across 128k vocab
        logits_f32 = logits.view(-1, logits.shape[-1])[-1].float()
        log_probs = torch.log_softmax(logits_f32, dim=-1)
        probs = torch.exp(log_probs)
        
        # H(p) in bits = - sum(p * ln(p)) / ln(2)
        entropy = -(probs * log_probs).sum().item() / 0.6931471805599453
        return max(0.0, float(entropy))

    def update_and_decide(self, logits: torch.Tensor) -> bool:
        step_entropy = self.compute_token_entropy(logits)
        
        if self.running_entropy == 0.0:
            self.running_entropy = step_entropy
        else:
            self.running_entropy = self.alpha * step_entropy + (1.0 - self.alpha) * self.running_entropy

        # Schmitt-trigger hysteresis state machine
        if self.speculation_active and self.running_entropy > self.theta_high:
            self.speculation_active = False  # Disengage draft model
        elif not self.speculation_active and self.running_entropy < self.theta_low:
            self.speculation_active = True   # Re-engage draft model

        return self.speculation_active