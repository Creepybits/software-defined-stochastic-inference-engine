# SDSIE: Software-Defined Stochastic Inference Engine  

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21499379.svg)](https://doi.org/10.5281/zenodo.21499379)
[![Hardware](https://img.shields.io/badge/Verified%20On-NVIDIA%20RTX%205090%20Blackwell-10b981.svg)](https://sdsie.github.io/)
[![Live Portal](https://img.shields.io/badge/Interactive%20Portal-sdsie.github.io-a855f7.svg)](https://sdsie.github.io/)

**Status: Corrected, component-level validation (August 2026).**
This repository was substantially revised on 2026-08-29 after independent re-verification found that
several headline numbers in the original paper and README (kernel latency, end-to-end throughput,
energy reduction) could not be reproduced. Those claims have been corrected here. The previous
version remains available on the [`pre-correction`](../../tree/pre-correction) branch for transparency.

If you found this project via the original paper, GitHub Pages site, or Zenodo record: please see
[Corrections](#corrections-from-the-original-release) below before citing any figure from those sources.

## What this is

SDSIE is an open-source runtime layer exploring **energy-proportional LLM serving** through two
independent mechanisms:

1. A **Triton INT4 GEMM kernel** that keeps weights packed in 4-bit format in global memory and
   dequantizes on-chip, aiming to cut memory bus traffic.
2. An **entropy-gated speculative decoding controller** — a Schmitt-trigger hysteresis clutch that
   uses live Shannon entropy of model output logits to decide when speculative drafting is likely to
   pay off.

**These two mechanisms, plus real speculative decoding, are each independently validated in this
repo — but they are not yet wired together into a single accelerated serving path.** See
[Current Status](#current-status) for exactly what that means.

## Current Status

| Component | Status | Evidence |
|---|---|---|
| INT4 kernel — memory bandwidth reduction | ✅ **Validated**, real | 71.9–73.4% reduction, reproduced across 3 independent scripts |
| INT4 kernel — latency reduction | ⚠️ **Real, but modest** | ~4% faster than FP16 at batch size 1 (not the 63% originally claimed) |
| INT4 kernel — output correctness | ❌ **Not yet tested** | No comparison against FP16 output on real weights; synthetic weights only |
| Entropy clutch — computation | ✅ **Validated**, real | Line-by-line reviewed, matches paper's equations, reproducible across runs |
| Entropy clutch — driving real generation | ❌ **Not yet connected** | Every benchmarked script computes a decision but does not act on it (see below) |
| Speculative decoding (scout→target) | ✅ **Validated**, real | Up to 1.81× speedup at 85.4% accept rate, 100% output fidelity, N=10 trials/prompt |
| Calibration (real checkpoint → packed weights) | ❌ **Does not exist yet** | Kernel is only tested against synthetic random weights |
| End-to-end integrated server | ❌ **Not yet built** | No script combines quantization + entropy gating + real speculation in one running path |

### The core finding

Nine scripts in this repository's history compute a live entropy-gated decision (`k_draft`,
`gear`, etc.) from real model logits. **Only one of them (`SDSIEDynamicLinear`, used for the kernel
benchmark) actually branches computation on that decision.** Every other script — including the
reference server — logs the decision and then performs identical work regardless of it. We
confirmed this directly: across 9 threshold configurations spanning 0–34% "speculative-gate active"
time, measured throughput stayed flat (51.1–51.8 tok/s), which is only possible if the gate isn't
affecting execution.

This is not a subtle bug — it's the main remaining engineering task. Connecting the validated clutch
to the validated branching mechanism and the validated speculative loop is what's left before this
project can report a genuine end-to-end number.

## Corrections from the original release

| Metric | Originally claimed | Corrected | Why |
|---|---|---|---|
| Kernel latency | 28.60 µs (–63.1%) | 76.49 µs (–4.0%) | Never reproduced by any script in this repo; real branching test shows a much smaller effect |
| Memory bandwidth | 75.0% | 71.9–73.4% | Close to original claim; minor correction |
| End-to-end throughput | 50.52 tok/s ("speculative") | 50.13 tok/s (plain generation) | Same script produces this number, but it does not perform real speculation (see Current Status) |
| Energy reduction | 46.7% (6.40→3.41 J) | 32.5%–60.0%, task-dependent (real, N≥10, fidelity-checked) | Original figure unreproducible; real measurement shows reduction scales with draft-acceptance rate rather than being a fixed constant |

Full details and real telemetry are in [`sdsie_paper.tex`](./sdsie_paper.tex) (build with
`pdflatex sdsie_paper.tex` — run twice) and in `tools/telemetry/`, where every number above is
traceable to a raw JSON/CSV file and the script that produced it.

## Repository structure

```
vllm_sdsie/
  kernels/entropy_clutch.py       - Schmitt-trigger entropy clutch (validated)
  spec_decode/sdsie_speculator.py - Thin controller wrapper (validated)
  quantization/                   - Empty; no calibration script exists yet
step3_speculative_scout.py        - Real scout->target speculative decoding
benchmark_academic_validation_v4.py - N=10 baseline-vs-speculative ablation
step2_triton_dynamic.py           - Real branching INT4/FP16 kernel benchmark
sweep_real_model.py               - Clutch behavior across theta configurations
sdsie_server.py                   - Reference server (clutch computed, not enacted)
tools/
  cognitive_benchmark.py          - Per-category energy/gear telemetry
  harness_telemetry.py            - Per-token live entropy/gear trace
  web_ui.py                       - Minimal browser UI for the reference server
  plot_*.py                       - Generates the figures in the paper
  telemetry/                      - Raw JSON/CSV output from every run above
  assets/                         - Generated plots (PNG)
```

## Running the benchmarks

Each script writes its output to `tools/telemetry/` under a unique filename. None of them require
each other except where noted (`sdsie_speculator.py` depends on `entropy_clutch.py`).

```bash
# Real speculative decoding ablation (takes several minutes, loads two models)
python3 benchmark_academic_validation_v4.py

# Real branching INT4 kernel benchmark (fast, no model loading)
python3 step2_triton_dynamic.py

# Reference server + entropy trace (start server, then send a request)
python3 sdsie_server.py
# in a second terminal:
python3 tools/web_ui.py
```

## What we'd ask readers to take from this

The underlying ideas — entropy-gated adaptive speculation, and this general approach — aren't
unclaimed territory; published work like AdaEDL (Qualcomm, 2024) and SGLang's adaptive speculative
length show the general technique can pay off elsewhere. What isn't yet demonstrated here is that
*this specific implementation* delivers it end-to-end. We'd rather report that honestly, with
reproducible partial results, than repeat an unreproducible headline number.

## Citation

See [`sdsie_paper.tex`](./sdsie_paper.tex) for the current BibTeX entry. The DOI-archived Zenodo
record will be updated to point to this corrected version.

## License

Apache-2.0.
