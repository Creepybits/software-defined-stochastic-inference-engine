"""
cognitive_fidelity_check.py

Fast fidelity-only iteration tool for the entropy-gated INT4/FP16 gear.
Deliberately skips two expensive things cognitive_benchmark.py does:
  - The multi-minute closed-loop thermal warmup.
  - The N-trial timed suite with power/energy measurement.

Neither is needed here. Token-level fidelity is a deterministic property of
the weights and the decoding math -- GPU clock/thermal state affects how
FAST a computation runs, not what numeric VALUE it produces. So a run right
after cold model load produces exactly the same token sequence as one after
20 minutes of warmup; there is nothing for a thermal warmup to fix here.
That means this script goes from model load to a fidelity answer for all
three reference prompts in about as long as it takes to generate
~6 x max_tokens tokens, not several minutes of warmup plus 30 timed trials.

Use this to quickly try different gating configurations (which projection,
how many layers) before committing to a full cognitive_benchmark.py run --
e.g. to test whether fidelity loss comes from quantizing many layers at
once (errors compounding through the residual stream across depth) vs. the
calibration scheme itself being too coarse for down_proj specifically.

Example:
    python cognitive_fidelity_check.py                    # all 32 layers
    python cognitive_fidelity_check.py --num-layers 4      # just the first 4
    python cognitive_fidelity_check.py --num-layers 1      # just one
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import bench_common
from vllm_sdsie.kernels.entropy_clutch import SchmittTriggerEntropyClutch
from vllm_sdsie.quantization.gated_linear import GearState, install_gated_layers

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:0"


def gated_generate(model, input_ids, clutch, gear_state, max_tokens, eos=None):
    """Same accept-no-verification greedy loop as cognitive_benchmark.py's
    gated_generate. Duplicated rather than imported so this script stays
    standalone and quick to read on its own -- if the decode loop itself
    ever needs a fix, apply it in both places.
    """
    current_ids = input_ids
    generated = []
    with torch.inference_mode():
        for _ in range(max_tokens):
            out = model(current_ids)
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = next_token.item()
            generated.append(token_id)

            active, _h_step, _h_ema = clutch.update_and_decide(logits)
            gear_state.active = active

            if eos and token_id in eos:
                break
            current_ids = torch.cat([current_ids, next_token], dim=-1)
    return generated


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=bench_common.REFERENCE_MAX_TOKENS)
    ap.add_argument("--target-attr", default="down_proj",
                     help="Which mlp projection to gate (default: down_proj).")
    ap.add_argument("--num-layers", type=int, default=None,
                     help="Gate only the first N decoder layers instead of all 32. "
                          "Default: gate every layer.")
    args = ap.parse_args()

    print("=" * 78)
    print("FIDELITY-ONLY CHECK (no warmup, no power measurement, no timed trials)")
    print(f"Gating: mlp.{args.target_attr}, "
          f"{'all layers' if args.num_layers is None else f'first {args.num_layers} layer(s)'}")
    print("=" * 78)

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos = bench_common.eos_ids(tokenizer)

    print("[*] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    layer_indices = range(args.num_layers) if args.num_layers is not None else None
    gear_state = GearState(active=False)
    n_patched = install_gated_layers(model, gear_state, target_attr=args.target_attr,
                                      layer_indices=layer_indices)
    print(f"[*] Patched {n_patched} layer(s) with real per-channel INT4 calibration.")

    clutch = SchmittTriggerEntropyClutch()
    print(f"[*] Clutch: theta_low={clutch.theta_low}, theta_high={clutch.theta_high}, alpha={clutch.alpha}\n")

    encoded = [(label, bench_common.encode_prompt(tokenizer, text, device=DEVICE))
               for label, text in zip(bench_common.REFERENCE_PROMPT_LABELS, bench_common.REFERENCE_PROMPTS)]

    for label, ids in encoded:
        gear_state.active = False
        base = bench_common.target_only_generate(model, ids, args.tokens, eos=eos)["generated_ids"]

        gear_state.active = False  # start safe -- see GearState docstring
        clutch.reset()
        gated = gated_generate(model, ids, clutch, gear_state, args.tokens, eos=eos)

        common_len = min(len(base), len(gated))
        match = sum(1 for a, b in zip(base[:common_len], gated[:common_len]) if a == b)
        match_pct = (match / common_len * 100.0) if common_len else 0.0
        first_div = next((i for i in range(common_len) if base[i] != gated[i]), None)
        div_str = f"first divergence at token {first_div}" if first_div is not None else "no divergence"
        print(f"  {label:<8}: {match_pct:5.1f}% match  {div_str}  ({common_len} tokens compared)")

    print("=" * 78)


if __name__ == "__main__":
    main()
