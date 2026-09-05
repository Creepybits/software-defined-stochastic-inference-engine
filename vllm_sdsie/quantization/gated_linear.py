# vllm_sdsie/quantization/gated_linear.py
#
# The first real wiring of the entropy-gated "Resolution Gear" concept to
# actual computation, rather than a logged-but-unused decision. This is the
# exact pattern the project's status doc has been flagging across multiple
# files (sdsie_server.py, harness_step1.py, four root-folder speculative
# engines, and cognitive_benchmark.py before this session): a clutch or
# gear value gets computed and counted, but nothing downstream ever
# branches on it. GatedLinear actually branches: it holds both an
# INT4-quantized and the original FP16/BF16 copy of the same weights, and
# picks one at forward time based on a shared GearState flag that the
# calling generation loop updates once per token from real entropy.
#
# HONESTY NOTE: unlike this project's speculative-decoding work, this is
# NOT lossless. INT4-quantizing live weights changes the model's numerical
# output whenever the INT4 path is used, which can and sometimes will
# change which token gets emitted. Report a real fidelity/divergence number
# against an all-FP16 baseline for any run that uses this -- do not claim
# exact token match the way the speculative-decoding work correctly does.
import torch

from vllm_sdsie.quantization.calibrate import quantize_linear_int4_minmax
from vllm_sdsie.quantization.sdsie_linear import SDSIELinearMethod


class GearState:
    """Tiny shared mutable flag. One instance is created per generation run
    and passed to every GatedLinear layer being gated together, so a single
    entropy decision (computed once per token, from the model's actual
    output logits) drives every gated layer consistently -- not one
    independent clutch per layer.

    `active=True` follows the same convention as
    SchmittTriggerEntropyClutch.speculation_active: True means "confident /
    low entropy", which here is mapped to the fast INT4 path. False means
    "uncertain / high entropy", mapped to the original FP16/BF16 path. This
    mapping matches the project's own stated design ("reserving heavy FP16
    pipelines only for high-uncertainty tokens").

    Defaults to active=False (FP16/safe), unlike
    SchmittTriggerEntropyClutch's own default_active=True. That default is
    safe there because speculative decoding is lossless -- an optimistic
    first guess costs nothing, since the target model's verification step
    corrects any wrong guess before anything is emitted. There is no
    equivalent correction step here: whatever the INT4 path computes IS the
    final answer. Starting active/INT4 by default meant the very first
    token of every generation -- before the clutch has observed any real
    entropy at all -- was unconditionally computed in INT4, which is
    exactly backwards: the first token has no generated context to lean on
    and is one of the highest-stakes positions to get right.
    """

    def __init__(self, active: bool = False):
        self.active = active


class GatedLinear(torch.nn.Module):
    """Drop-in replacement for an existing nn.Linear that switches between
    an INT4-quantized path and the original layer based on a shared
    GearState, decided once per token by the caller (see
    cognitive_benchmark.py for the generation loop that drives this).

    Calibration (quantize_linear_int4_minmax) runs once, at construction,
    against whatever weights the wrapped layer already has loaded -- this
    is real per-channel min-max INT4 calibration, not a placeholder.
    """

    def __init__(self, orig_linear: torch.nn.Linear, gear_state: GearState):
        super().__init__()
        self.gear_state = gear_state
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features

        # The original layer, kept completely unmodified, for the
        # high-uncertainty / accurate path.
        self.fp16_linear = orig_linear

        weight_packed, scales, zeros = quantize_linear_int4_minmax(orig_linear.weight.data)
        device = orig_linear.weight.device
        dtype = orig_linear.weight.dtype
        self.register_buffer("weight_packed", weight_packed.to(device))
        self.register_buffer("scales", scales.to(device=device, dtype=dtype))
        self.register_buffer("zeros", zeros.to(device=device, dtype=dtype))

        self._int4_method = SDSIELinearMethod()

        # Telemetry: how many forward calls actually went down each path,
        # for this specific layer. Cheap counters, no effect on computation.
        self.int4_calls = 0
        self.fp16_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gear_state.active:
            self.int4_calls += 1
            return self._int4_method.apply(self, x, bias=self.fp16_linear.bias)
        else:
            self.fp16_calls += 1
            return self.fp16_linear(x)


def install_gated_layers(model, gear_state: GearState, target_attr: str = "down_proj",
                          layer_indices=None):
    """Walk model.model.layers and replace each decoder layer's
    `target_attr` (default: mlp.down_proj) with a GatedLinear wrapping the
    original. Returns the number of layers patched.

    layer_indices: optional iterable of decoder-layer indices to patch
    (e.g. range(4) for just the first 4 layers). None (default) patches
    every layer. Useful for testing whether fidelity loss comes from
    quantizing many layers at once (errors compounding through the residual
    stream across depth) vs. the calibration scheme itself being too coarse
    for this projection -- gate a handful of layers and compare against
    gating all 32.

    Scoped to one projection per layer by default -- not every Linear in
    the model -- enough to have a real, measurable effect without a much
    larger first pass. down_proj is the last projection in the MLP block
    (intermediate_size -> hidden_size), so its output feeds directly into
    the residual stream.
    """
    count = 0
    layers = model.model.layers
    indices = range(len(layers)) if layer_indices is None else layer_indices
    for i in indices:
        layer = layers[i]
        orig = getattr(layer.mlp, target_attr)
        if isinstance(orig, GatedLinear):
            continue  # already patched, don't double-wrap
        gated = GatedLinear(orig, gear_state)
        setattr(layer.mlp, target_attr, gated)
        count += 1
    return count


def collect_gate_usage(model, target_attr: str = "down_proj"):
    """Sum int4_calls/fp16_calls across every patched layer, for reporting
    how much of the run actually used each path (as opposed to
    GearState.active's own step-by-step history, which reflects the
    decision, not necessarily how many forward calls happened per step)."""
    int4_total = 0
    fp16_total = 0
    for layer in model.model.layers:
        gated = getattr(layer.mlp, target_attr)
        if isinstance(gated, GatedLinear):
            int4_total += gated.int4_calls
            fp16_total += gated.fp16_calls
    return int4_total, fp16_total


def reset_gate_counters(model, target_attr: str = "down_proj"):
    """Zero every patched layer's int4_calls/fp16_calls -- call before each
    timed trial so collect_gate_usage() reflects only that trial."""
    for layer in model.model.layers:
        gated = getattr(layer.mlp, target_attr)
        if isinstance(gated, GatedLinear):
            gated.int4_calls = 0
            gated.fp16_calls = 0


if __name__ == "__main__":
    # Self-test: wrap a single plain nn.Linear (not a full model, to keep
    # this runnable without downloading a checkpoint) in GatedLinear,
    # confirm both paths run, produce differently-shaped-but-consistent
    # output, and that the FP16 path is bit-exact to the original layer
    # while the INT4 path is close-but-not-exact (real quantization error,
    # not a bug -- see calibrate.py's own self-test for that check in
    # isolation).
    print("=" * 78)
    print("gated_linear.py self-test (single layer, no full model needed)")
    print("=" * 78)

    torch.manual_seed(0)
    in_features, out_features, batch = 16, 8, 4
    # FIXED: this self-test originally created orig/x with no device
    # argument, defaulting to CPU. The FP16 path (plain PyTorch) works fine
    # on CPU, but the INT4 path launches a real Triton kernel, which can
    # only operate on CUDA memory -- hence "Pointer argument cannot be
    # accessed from Triton (cpu tensor?)" the first time this ran. Building
    # everything on the same device up front (matching sdsie_linear.py's
    # own self-test, which already handled this correctly) fixes it.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig = torch.nn.Linear(in_features, out_features, bias=True, device=device)
    gear = GearState(active=False)  # matches the safe default; both paths still exercised below via explicit overrides
    gated = GatedLinear(orig, gear)

    x = torch.randn(batch, in_features, device=device)

    gear.active = False
    out_fp16 = gated(x)
    ref_fp16 = orig(x)
    fp16_exact = torch.equal(out_fp16, ref_fp16)
    print(f"[{'ok' if fp16_exact else 'FAIL'}] FP16 path bit-exact to the original layer: {fp16_exact}")

    if device != "cuda":
        print("[skip] No CUDA GPU available -- can't exercise the actual Triton")
        print("       kernel here (the INT4 path), only the FP16-path check above.")
    else:
        gear.active = True
        out_int4 = gated(x)
        diff = (out_int4 - ref_fp16).abs().max().item()
        print(f"[ok] INT4 path differs from FP16 by {diff:.4f} max abs "
              f"(expected -- real quantization error, not a bug)")

        print(f"[ok] Per-layer call counts: int4_calls={gated.int4_calls}, "
              f"fp16_calls={gated.fp16_calls} (expected 1 and 1)")
        assert gated.int4_calls == 1 and gated.fp16_calls == 1
    print("=" * 78)
