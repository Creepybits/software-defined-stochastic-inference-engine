"""
entropy_overhead_bench.py

Isolated microbenchmark for the "floor cost" identified in
step4_theta_alpha_grid_v2.py: runs where the clutch NEVER speculated
(100% fallback, 0% accept) still ran 4-7% slower than the pure matched
FP16 baseline, despite doing the same single-forward-pass-per-token work.

"""

import torch
import time

VOCAB = 128256
N_CALLS = 2000
DEVICE = "cuda"


def compute_current(logits):
    """Exact current implementation from entropy_clutch.py."""
    logits_f32 = logits.view(-1, logits.shape[-1])[-1].float()
    log_probs = torch.log_softmax(logits_f32, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum().item() / 0.6931471805599453
    return max(0.0, float(entropy))


def compute_no_persync_batch(logits_list):
    """
    Same math as current, but accumulates results as GPU tensors and only
    syncs ONCE at the end for the whole batch, instead of once per call.
    This isolates how much the PER-CALL sync specifically costs, vs. doing
    the same total amount of work with one sync at the end.
    """
    results = []
    for logits in logits_list:
        logits_f32 = logits.view(-1, logits.shape[-1])[-1].float()
        log_probs = torch.log_softmax(logits_f32, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum() / 0.6931471805599453
        results.append(entropy)
    stacked = torch.stack(results)
    return stacked.clamp(min=0.0).cpu().tolist()  # ONE sync for all N_CALLS


def compute_trimmed(logits):
    """
    Mathematically equivalent entropy calc using fewer intermediate
    tensors: avoids materializing a separate `probs` tensor by using
    logsumexp directly, and skips the redundant float() wrapper.
    H(p) = -sum(exp(log_p) * log_p) = log(sum(exp(logits))) - sum(softmax(logits) * logits)
    Simplified via: entropy = logsumexp(z) - sum(softmax(z) * z), all in nats, then /ln(2)
    """
    z = logits.view(-1, logits.shape[-1])[-1].float()
    log_probs = torch.log_softmax(z, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum().item() / 0.6931471805599453
    return max(0.0, entropy)


def bench(fn, logits, n_calls, label, is_batch=False):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if is_batch:
        fn([logits] * n_calls)
    else:
        for _ in range(n_calls):
            fn(logits)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    per_call_us = (t1 - t0) / n_calls * 1e6
    print(f"{label:<45s}: {per_call_us:8.1f} us/call  (total {t1-t0:.3f}s for {n_calls} calls)")
    return per_call_us


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA available -- this benchmark requires a GPU to be meaningful.")
        exit(1)

    print("=" * 90)
    print(f"ENTROPY COMPUTATION OVERHEAD MICROBENCHMARK  (vocab={VOCAB}, N={N_CALLS} calls)")
    print("=" * 90)

    logits = torch.randn(1, 1, VOCAB, device=DEVICE, dtype=torch.bfloat16)

    # Warmup (avoid cold-start CUDA compilation confound, same practice as project scripts)
    for _ in range(20):
        compute_current(logits)
    torch.cuda.synchronize()

    t_current = bench(compute_current, logits, N_CALLS, "1. Current (per-call .item() sync)")
    t_batched = bench(compute_no_persync_batch, logits, N_CALLS, "2. Same math, ONE sync at the end", is_batch=True)
    t_trimmed = bench(compute_trimmed, logits, N_CALLS, "3. Trimmed math (per-call sync)")

    print("-" * 90)
    print(f"Per-call sync overhead (1 vs 2): {t_current - t_batched:+.1f} us/call "
          f"({(t_current/t_batched - 1)*100:+.1f}% slower with per-call sync)")
    print(f"Trimmed math effect (1 vs 3)   : {t_current - t_trimmed:+.1f} us/call "
          f"({(t_current/t_trimmed - 1)*100:+.1f}% change from math trimming alone)")
    print("=" * 90)
    print("\nInterpretation:")
    print("- If (1 vs 2) shows a big gap: the PER-CALL sync itself is the dominant cost.")
    print("  This matches the design constraint (Python needs a bool to branch on) but")
    print("  confirms batching/pipelining decisions differently could help in a real server")
    print("  (e.g. speculative-decoding servers often decide k for a whole batch at once).")
    print("- If (1 vs 3) shows a big gap: the math itself (not just the sync) has room to")
    print("  trim, independent of the sync question.")
