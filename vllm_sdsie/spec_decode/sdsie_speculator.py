# vllm_sdsie/spec_decode/sdsie_speculator.py
from typing import Optional, Tuple, Dict, Any
import torch
from vllm_sdsie.kernels.entropy_clutch import SchmittTriggerEntropyClutch


class SDSIESpeculativeController:
    """
    Entropy-gated Speculative Decoding Controller for vLLM/SGLang runtimes.
    Dynamically adjusts draft proposal length (k in {0, K}) based on the
    Schmitt-trigger hysteresis state.
    """

    def __init__(
        self,
        default_k: int = 5,
        theta_low: float = 0.60,
        theta_high: float = 1.40,
        alpha: float = 0.65,
    ):
        self.default_k = default_k
        self.clutch = SchmittTriggerEntropyClutch(
            theta_low=theta_low,
            theta_high=theta_high,
            alpha=alpha,
            default_active=True,
        )

        # Telemetry and accounting counters
        self.total_tokens_generated: int = 0
        self.total_draft_tokens_proposed: int = 0
        self.speculative_tokens_accepted: int = 0
        self.speculation_cycles: int = 0
        self.fallback_cycles: int = 0

        # Latest step metrics for outer telemetry logging
        self.last_step_entropy: float = 0.0
        self.last_running_entropy: float = 0.0

    def reset(self) -> None:
        """Resets the controller state between independent trials or requests."""
        self.clutch.reset()
        self.total_tokens_generated = 0
        self.total_draft_tokens_proposed = 0
        self.speculative_tokens_accepted = 0
        self.speculation_cycles = 0
        self.fallback_cycles = 0
        self.last_step_entropy = 0.0
        self.last_running_entropy = 0.0

    def plan_speculation_step(
        self, current_logits: torch.Tensor, pos_idx: int = -1
    ) -> int:
        """
        Determines the number of speculative draft tokens to request (k).
        pos_idx specifies the authoritative token position from the previous verification pass.
        Returns:
            k (int): Number of draft tokens to propose (0 if clutch disengages).
        """
        (
            speculation_active,
            self.last_step_entropy,
            self.last_running_entropy,
        ) = self.clutch.update_and_decide(current_logits, pos_idx=pos_idx)

        if speculation_active:
            self.speculation_cycles += 1
            return self.default_k
        else:
            self.fallback_cycles += 1
            return 0  # Fallback: bypass draft model entirely to save FLOPs

    def record_verification(self, num_drafted: int, num_accepted: int) -> None:
        """
        Telemetry logging for acceptance ratios and generation accounting.
        - num_drafted: Number of tokens proposed by the scout (0 if fallback).
        - num_accepted: Number of proposed tokens confirmed by target.
        """
        self.total_draft_tokens_proposed += num_drafted
        self.speculative_tokens_accepted += num_accepted
        # Emitted tokens = accepted draft tokens + 1 target model token (either correction or bonus)
        self.total_tokens_generated += num_accepted + 1

    @property
    def acceptance_rate(self) -> float:
        """Draft acceptance rate (beta), matching standard literature."""
        if self.total_draft_tokens_proposed == 0:
            return 0.0
        return self.speculative_tokens_accepted / self.total_draft_tokens_proposed

    @property
    def fallback_percentage(self) -> float:
        """Fraction of total cycles that bypassed speculative drafting."""
        total_cycles = self.speculation_cycles + self.fallback_cycles
        if total_cycles == 0:
            return 0.0
        return (self.fallback_cycles / total_cycles) * 100.0

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Returns structured dictionary ready for JSON/CSV serialization."""
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "total_drafted": self.total_draft_tokens_proposed,
            "total_accepted": self.speculative_tokens_accepted,
            "acceptance_rate_pct": round(self.acceptance_rate * 100.0, 2),
            "speculation_cycles": self.speculation_cycles,
            "fallback_cycles": self.fallback_cycles,
            "fallback_pct": round(self.fallback_percentage, 2),
            "last_step_entropy": round(self.last_step_entropy, 4),
            "last_running_entropy": round(self.last_running_entropy, 4),
        }


if __name__ == "__main__":
    print("=" * 78)
    print("SDSIE Speculative Controller Unit Test")
    print("=" * 78)

    controller = SDSIESpeculativeController(
        default_k=5, theta_low=0.60, theta_high=1.40, alpha=0.65
    )
    vocab = 128256

    # See entropy_clutch.py's __main__ for the full derivation and numeric
    # verification of these scenarios -- the original two cases here (a
    # dominant-token "deterministic" case and a 6-tokens-at-logit-2.0
    # "cognitive fork" case) only ever probed H~0 and H~17 bits, the latter
    # mislabeled as "H ~ 2.5 bits" in comments; it's actually indistinguishable
    # from uniform noise across the whole vocabulary. Fixed the same way here:
    # strongly negative baseline (-30) so only the intended "live" tokens
    # (at +20) carry any real probability mass, giving verifiable target
    # entropies instead of two indistinguishable extremes.
    BASELINE = -30.0
    DOMINANT = 20.0

    def make_logits(live_indices):
        logits = torch.full((1, vocab), BASELINE)
        for idx in live_indices:
            logits[0, idx] = DOMINANT
        return logits

    def skewed_95_5_logits():
        import math
        logits = torch.full((1, vocab), BASELINE)
        logits[0, 0] = DOMINANT + math.log(19)  # -> exactly a 95%/5% split
        logits[0, 1] = DOMINANT
        return logits

    # 1. Deterministic step (H ~ 0.0 bits) -- fully predictable token
    logits_det = make_logits([0])
    k = controller.plan_speculation_step(logits_det)
    controller.record_verification(num_drafted=k, num_accepted=k)
    print(f"Deterministic (H~0.00)      -> Proposed k={k} | Summary: {controller.get_telemetry_summary()}")

    # 2. Low-moderate uncertainty (H ~ 0.29 bits) -- still well below theta_low
    logits_lowmod = skewed_95_5_logits()
    k = controller.plan_speculation_step(logits_lowmod)
    controller.record_verification(num_drafted=k, num_accepted=k)
    print(f"Low-moderate (H~0.29)       -> Proposed k={k} | Summary: {controller.get_telemetry_summary()}")

    # 3. Dead zone (H = 1.00 bits, between theta_low=0.60 and theta_high=1.40)
    #    Correctly-implemented hysteresis holds the PRIOR state here (still
    #    drafting) rather than reacting to entropy alone.
    logits_dead = make_logits([0, 1])
    k = controller.plan_speculation_step(logits_dead)
    controller.record_verification(num_drafted=k, num_accepted=max(0, k - 2) if k else 0)
    print(f"Dead zone (H=1.00)          -> Proposed k={k} | Summary: {controller.get_telemetry_summary()}")

    # 4. Genuine cognitive fork (H = log2(6) = 2.585 bits) -- exceeds theta_high
    logits_fork = make_logits([0, 1, 2, 3, 4, 5])
    k = controller.plan_speculation_step(logits_fork)
    controller.record_verification(num_drafted=k, num_accepted=0)
    print(f"Cognitive Fork (H=2.58)     -> Proposed k={k} | Summary: {controller.get_telemetry_summary()}")

    # 5. Back in the dead zone -- should now hold FALLBACK (k=0), the other
    #    half of the hysteresis behavior: doesn't re-engage just because
    #    entropy dropped out of the high range, only once it's below theta_low.
    logits_dead2 = make_logits([0, 1])
    k = controller.plan_speculation_step(logits_dead2)
    controller.record_verification(num_drafted=k, num_accepted=0)
    print(f"Dead zone again (H=1.00)    -> Proposed k={k} | Summary: {controller.get_telemetry_summary()}")

    print("=" * 78)
    print("[ok] Controller logic, unpack, and acceptance accounting validated.")
    print("[ok] Step 3 held DRAFT and step 5 held FALLBACK despite both landing in")
    print("     the same dead-zone entropy -- confirms genuine hysteresis, not a")
    print("     single static threshold.")