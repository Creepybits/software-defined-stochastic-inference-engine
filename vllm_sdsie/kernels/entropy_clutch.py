# vllm_sdsie/kernels/entropy_clutch.py
from typing import Optional, Tuple
import torch


class SchmittTriggerEntropyClutch:
    """
    Industrial Schmitt-Trigger Hysteresis Clutch for Speculative Decoding.
    Numerically stable for large-vocabulary LLMs (128k+ tokens).
    """

    def __init__(
        self,
        theta_low: float = 0.60,
        theta_high: float = 1.40,
        alpha: float = 0.65,
        default_active: bool = True,
    ):
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.alpha = alpha
        self.default_active = default_active

        self.running_entropy: Optional[float] = None
        self.last_step_entropy: float = 0.0
        self.speculation_active: bool = default_active

    def reset(self) -> None:
        """Resets running state between independent sequences or trials."""
        self.running_entropy = None
        self.last_step_entropy = 0.0
        self.speculation_active = self.default_active

    def compute_token_entropy(self, logits: torch.Tensor, pos_idx: int = -1) -> float:
        """
        Computes Shannon entropy: H(p) = -sum(p * log2(p)) in float32.
        pos_idx specifies which token position to evaluate (vital for verification batches).
        """
        # Ensure 2D [batch_or_seq, vocab] and select the authoritative emitted position
        flat_logits = logits.view(-1, logits.shape[-1])
        logits_f32 = flat_logits[pos_idx].float()

        # Numerically robust entropy calculation via torch.special.entr: -p * ln(p)
        probs = torch.softmax(logits_f32, dim=-1)
        entropy_nats = torch.special.entr(probs).sum().item()

        # Convert nats to Shannon bits (1 / ln(2) ≈ 1.4426950408889634)
        entropy_bits = entropy_nats * 1.4426950408889634
        return max(0.0, float(entropy_bits))

    def update_and_decide(
        self, logits: torch.Tensor, pos_idx: int = -1
    ) -> Tuple[bool, float, float]:
        """
        Updates the EMA state and evaluates the Schmitt-trigger hysteresis clutch.
        Returns: (speculation_active, step_entropy, running_entropy)
        """
        self.last_step_entropy = self.compute_token_entropy(logits, pos_idx=pos_idx)

        if self.running_entropy is None:
            self.running_entropy = self.last_step_entropy
        else:
            self.running_entropy = (
                self.alpha * self.last_step_entropy
                + (1.0 - self.alpha) * self.running_entropy
            )

        # Schmitt-trigger hysteresis state machine
        if self.speculation_active and self.running_entropy > self.theta_high:
            self.speculation_active = False  # Disengage draft model (fallback)
        elif not self.speculation_active and self.running_entropy < self.theta_low:
            self.speculation_active = True   # Re-engage draft model (speculate)

        return self.speculation_active, self.last_step_entropy, self.running_entropy


if __name__ == "__main__":
    print("=" * 78)
    print("SDSIE Schmitt-Trigger Entropy Clutch Self-Test")
    print("=" * 78)

    clutch = SchmittTriggerEntropyClutch(theta_low=0.60, theta_high=1.40, alpha=0.65)
    vocab_size = 128256  # Llama-3 vocabulary dimension
    max_possible_entropy = torch.log2(torch.tensor(float(vocab_size))).item()

    print(f"Config : theta_low={clutch.theta_low} bits, theta_high={clutch.theta_high} bits, alpha={clutch.alpha}")
    print(f"Vocab  : {vocab_size:,} tokens (max possible entropy: {max_possible_entropy:.3f} bits)\n")

    # ------------------------------------------------------------------
    # Test scenario construction.
    #
    # FIXED (was buggy): the original "cognitive fork" case set 6 tokens to
    # logit=2.0 against a baseline of 0.0 for the other ~128,250 tokens,
    # commented as producing "H ~ 2.5 bits". Measured, it actually produces
    # ~16.97 bits -- essentially indistinguishable from uniform noise across
    # the *entire* vocabulary -- because a logit gap of only 2.0 is nowhere
    # near enough to suppress a tail of 128,250 other tokens. Both self-tests
    # in this file and in sdsie_speculator.py only ever probed the two
    # extremes (H~0 and H~max); neither could tell you anything about
    # whether theta_low/theta_high are sensible, since virtually any pair of
    # thresholds between 0 and ~17 would "pass" trivially.
    #
    # Fixed by giving every non-target token a strongly negative baseline
    # logit (-30) and setting only the intended "live" tokens to a large,
    # equal value (20) -- a gap of 50 is large enough that the tail's total
    # contribution to entropy is below 1e-10 bits, so each case's measured
    # entropy matches its closed-form target to 4+ decimal places (verified
    # numerically before writing this). This gives four genuinely distinct,
    # verifiable entropy regions instead of two indistinguishable extremes:
    #   - Deterministic:        1 dominant token           -> H ~ 0.000 bits
    #   - Low-moderate (95/5):  2 tokens, skewed 95%/5%     -> H ~ 0.286 bits
    #   - Dead zone (50/50):    2 tokens, equal split       -> H = 1.000 bits
    #   - High entropy (fork):  6 tokens, equal split       -> H = log2(6) = 2.585 bits
    # ------------------------------------------------------------------

    BASELINE = -30.0
    DOMINANT = 20.0  # gap of 50 vs. baseline -> tail contribution negligible (<1e-10 bits)

    def make_logits(live_indices: list) -> torch.Tensor:
        logits = torch.full((1, vocab_size), BASELINE)
        for idx in live_indices:
            logits[0, idx] = DOMINANT
        return logits

    def skewed_95_5_logits() -> torch.Tensor:
        # log(0.95/0.05) = log(19) -- the logit gap between the two tokens
        # that reproduces exactly a 95%/5% probability split.
        import math
        logits = torch.full((1, vocab_size), BASELINE)
        logits[0, 0] = DOMINANT + math.log(19)
        logits[0, 1] = DOMINANT
        return logits

    scenarios = [
        ("Deterministic",           make_logits([0])),
        ("Deterministic",           make_logits([0])),
        ("Low-moderate (95/5)",     skewed_95_5_logits()),
        ("Low-moderate (95/5)",     skewed_95_5_logits()),
        ("Dead zone (50/50)",       make_logits([0, 1])),
        ("High entropy (6-way fork)", make_logits([0, 1, 2, 3, 4, 5])),
        ("Dead zone (50/50)",       make_logits([0, 1])),
        ("Low-moderate (95/5)",     skewed_95_5_logits()),
        ("Deterministic",           make_logits([0])),
    ]

    print(f"{'Step':<6} | {'Scenario':<26} | {'H_step (bits)':<14} | {'H_ema (bits)':<14} | {'Clutch State':<14}")
    print("-" * 78)
    for i, (label, logits) in enumerate(scenarios, 1):
        active, h_step, h_ema = clutch.update_and_decide(logits)
        state_str = "[k=5 DRAFT]" if active else "[k=0 FALLBACK]"
        print(f"{i:<6} | {label:<26} | {h_step:<14.4f} | {h_ema:<14.4f} | {state_str:<14}")

    print("-" * 78)
    print(f"[ok] Disengages on H_ema > {clutch.theta_high:.2f}, re-engages on H_ema < {clutch.theta_low:.2f}.")
    print("[ok] Steps 5 and 7 (dead zone, between theta_low and theta_high) hold their")
    print("     PRIOR state rather than reacting to the raw entropy value directly --")
    print("     this is the actual hysteresis behavior, not just a single threshold.\n")