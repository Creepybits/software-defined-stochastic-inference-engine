# test_spec_controller.py
import torch
from vllm_sdsie import SDSIESpeculativeController

def test_controller():
    print("=" * 60)
    print("Testing SDSIE Speculative Controller...")
    print("=" * 60)

    controller = SDSIESpeculativeController(default_k=5, theta_low=0.55, theta_high=0.82)
    vocab_size = 32000

    # Step 1: Predictable prompt (Sharp distribution)
    logits_low = torch.randn((1, vocab_size), device="cuda:0")
    logits_low[0, 100] = 40.0
    k1 = controller.plan_speculation_step(logits_low)
    print(f"Step 1 (Confident): Draft proposal k = {k1} (Expected: 5)")

    # Step 2: Unpredictable reasoning fork (Flat distribution)
    logits_high = torch.randn((1, vocab_size), device="cuda:0") * 0.01
    k2 = controller.plan_speculation_step(logits_high)
    print(f"Step 2 (Uncertain): Draft proposal k = {k2} (Expected: 0 - Clutch disengaged)")

    assert k1 == 5 and k2 == 0, "Controller failed to dynamically switch k!"
    print("✓ Speculative Controller passed dynamic draft length validation!")

if __name__ == "__main__":
    test_controller()