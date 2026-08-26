
# Software-Defined Stochastic Inference Engine (SDSIE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21499379.svg)](https://doi.org/10.5281/zenodo.21499379)
[![Hardware](https://img.shields.io/badge/Verified%20On-NVIDIA%20RTX%205090%20Blackwell-10b981.svg)](https://sdsie.github.io/)
[![Live Portal](https://img.shields.io/badge/Interactive%20Portal-sdsie.github.io-a855f7.svg)](https://sdsie.github.io/)

Official repository and research specification for the **Software-Defined Stochastic Inference Engine (SDSIE)**.

<p align="center">
  <img src="assets/sdsie-hero.png" alt="SDSIE Memory Wall Breakthrough on Blackwell" width="100%">
</p>

> 📄 **[Read the Full Research Paper (PDF)](https://doi.org/10.5281/zenodo.21499379)** | 🌐 **[Interactive Datacenter ROI Calculator](https://sdsie.github.io/)**

---

### Key Empirical Telemetry (Bare-Metal NVIDIA RTX 5090 Blackwell)

| Metric | Baseline (Static FP16/BF16) | SDSIE (Dynamic Sub-Byte INT4) | Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Energy Consumption** | 6.40 J / token | **3.41 J / token** | **–46.7% Energy Reduction** |
| **Memory Bus Traffic** | 117.44 MB / layer | **33.03 MB / layer** | **–71.9% Bus Traffic Cut** |
| **Kernel Latency** | 73.50 µs | **74.22 µs** | **+0.9% (Zero-Stall SRAM)** |
| **Throughput Velocity** | 23.70 tok/s | **34.91 tok/s** | **+47.3% Acceleration** |  

---

## 🔬 Reproducibility & Benchmark Quickstart

To reproduce the bare-metal kernel latency and power telemetry on your local GPU (NVIDIA Ampere, Ada Lovelace, Hopper, or Blackwell):

### 1. Environment Setup
```bash
git clone https://github.com/Creepybits/software-defined-stochastic-inference-engine.git
cd software-defined-stochastic-inference-engine
pip install torch triton nvidia-ml-py pynvml
```
### 2. Run Benchmarks

#### A. Isolated Triton SRAM Kernel Micro-Benchmark (Fast, 2s run)
```bash
python benchmark_telemetry.py
```

### B. End-to-End Autoregressive Model Profiler (Full Llama-3.1-8B)
```bash
python harness_telemetry.py meta-llama/Llama-3.1-8B-Instruct
```

## 📦 Drop-In vLLM Plugin Architecture (`vllm-sdsie`)

SDSIE includes an installable, modular runtime plugin for vLLM and open-source inference servers featuring fused sub-byte Triton GEMM kernels and Schmitt-trigger speculative decoding control:

### 1. Installation
```bash
# Clone and install SDSIE in editable mode
git clone https://github.com/Creepybits/software-defined-stochastic-inference-engine.git
cd software-defined-stochastic-inference-engine
pip install -e .
```
### 2. Standalone Verification Suite
* Micro-Kernel Sanity Check:
```bash
python test_vllm_sdsie.py
```
Telemetry: 28.6 µs isolated GEMM latency on RTX 5090 Blackwell.

* Full Llama-3-8B SwiGLU MLP Block Forward Pass:
```bash
python test_transformer_block.py
```
Telemetry: 155.8 µs layer latency (–75.0% global VRAM memory bus traffic reduction).

* Entropy-Gated Speculative Controller:
```bash
python test_spec_controller.py
```
Telemetry: Dynamic draft scaling (k=5 confident → k=0 uncertain fallback).

### 3. Runtime Integration
```bash
import vllm_sdsie

# Dynamically hooks SDSIE sub-byte kernels into the active engine runner
vllm_sdsie.patch_vllm()
```
---

## 📊 Comprehensive Hardware Telemetry Matrix

The 4-panel telemetry matrix below illustrates empirical measurements captured across varied cognitive workloads and speculative engine configurations on bare-metal **NVIDIA Blackwell (RTX 5090 32GB)** under continuous 100 Hz NVML power polling:

<p align="center">
  <img src="assets/sdsie_telemetry_matrix.png" alt="SDSIE Empirical Telemetry Matrix" width="100%">
</p>

---

### Key Architectural Takeaways:
1. **Thermodynamic Efficiency:** Speculative scouting drops energy consumption to **3.41 J / token** compared to standard autoregressive baseline.
2. **Information Compressibility:** Code generation achieves a **94.7% sub-byte INT4 duty cycle**, validating near-zero syntactic entropy.
3. **Throughput Scaling:** Fast Speculative (Pure KV) reaches **34.91 tok/s** end-to-end generation velocity.
4. **Power Dissipation Profile:** Average active power hovers around **119.1 W to 147.2 W** during speculative high-gear execution compared to >330 W unthrottled loads.

## Author
* **Zanno Jacklin** ([Creepybits](https://zanno.se))
* Date: August 2026
