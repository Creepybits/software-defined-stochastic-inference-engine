# Software-Defined Stochastic Inference Engine (SDSIE)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21499379.svg)](https://doi.org/10.5281/zenodo.21499379)
[![Hardware](https://img.shields.io/badge/Verified%20On-NVIDIA%20RTX%205090%20Blackwell-10b981.svg)](https://sdsie.github.io/)
[![Live Portal](https://img.shields.io/badge/Interactive%20Portal-sdsie.github.io-a855f7.svg)](https://sdsie.github.io/)

Official repository and research specification for the **Software-Defined Stochastic Inference Engine (SDSIE)**.

<p align="center">
  <img src="assets/sdsie-hero.png" alt="SDSIE Memory Wall Breakthrough on Blackwell" width="100%">
</p>

> 📄 **[Read the IEEE Research Paper (PDF)](https://doi.org/10.5281/zenodo.22129912)** | 🌐 **[Interactive Datacenter ROI & CO₂ Calculator](https://sdsie.github.io/)**

---

### ⚡ Key Empirical Telemetry (Bare-Metal NVIDIA RTX 5090 Blackwell)

All metrics verified on an isolated bare-metal **NVIDIA GeForce RTX 5090 32GB (Blackwell Architecture)** via continuous 100 Hz NVML hardware power polling on **Llama-3.1-8B**:

| Metric | Baseline (Static FP16/BF16) | SDSIE (Dynamic Sub-Byte INT4) | Impact / Hardware Delta |
| :--- | :---: | :---: | :---: |
| **Electrical Energy / Token** | $6.40\text{ J}$ | **$3.41\text{ J}$** | **–46.7% Energy Reduction** |
| **Memory Bus Traffic / Layer** | $117.44\text{ MB}$ | **$29.36\text{ MB}$** | **–75.0% Bus Traffic Cut** |
| **Isolated GEMM Latency** | $77.46\ \mu\text{s}$ | **$28.60\ \mu\text{s}$** | **–63.1% Kernel Speedup** |
| **SwiGLU MLP Block Latency** | $622.4\ \mu\text{s}$ | **$155.8\ \mu\text{s}$** | **–74.9% Forward Pass Latency** |
| **Sustained Real Throughput** | $23.70\text{ tok/s}$ | **$50.52\text{ tok/s}$** | **+113.2% Speedup ($2.13\times$)** |
| **Active Core Power** | $142.1\text{ W}$ | **$95.3\text{ W}$** | **–32.9% Active Power Draw** |

---

## 🧠 Architectural Innovations

Autoregressive decoding in LLMs is fundamentally bounded by the memory bandwidth wall ($\approx 1\text{ FLOP/byte}$). SDSIE treats LLM inference as a dynamic thermodynamic control problem via two complementary subsystems:

1. **Fused Sub-Byte SRAM Dequantization:** Custom OpenAI Triton GEMM kernels stream packed INT4 weights from global VRAM and execute register-fused expansion to half-precision directly within Streaming Multiprocessor (SM) SRAM, eliminating 75.0% of interconnect traffic.
2. **Schmitt-Trigger Stochastic Clutch:** A real-time control loop continuously computes output distribution Shannon entropy in float32 space ($\theta_{\text{low}}=0.55$, $\theta_{\text{high}}=1.25$, EMA $\alpha=0.35$). It dynamically engages speculative draft verification ($k=5$) during deterministic text and drops to single-step fallback ($k=0$) during high-entropy cognitive forks, eliminating draft-rejection compute waste.

---

## 📊 Empirical Telemetry & Parameter Sensitivity

### 1. Real-Time Stochastic Speculation Trace (*Chant Royal*)
Below is a live 409-token telemetry trace of **Llama-3.1-8B** executing on an RTX 5090 Blackwell workstation under strict medieval poetic structure:

<p align="center">
  <img src="./assets/sdsie_chant_royal_trace.png" alt="SDSIE Stochastic Telemetry Trace" width="100%">
</p>

* **Deterministic Cadence ($H_{\text{min}} = 0.0001\text{ bits}$):** During rhyming couplets (Steps 90–240), the clutch locked at $k=5$ for over 150 consecutive tokens with $\approx 100\%$ draft acceptance.
* **Cognitive Rhyme Search ($H_{\text{max}} = 2.98\text{ bits}$):** During multi-stanza structural rhyme turns, entropy spiked above $1.25\text{ bits}$, triggering instant single-step fallback ($k=0$) and avoiding wasted draft compute.
* **Result:** 409 tokens completed in $8.10\text{s}$ at **$50.52\text{ tok/s}$** and **$3.41\text{ J/token}$**.

<details>
<summary>📜 <b>Click to expand full 409-token Chant Royal output</b></summary>

> **Prompt:** `Write an original Chant Royal poem in English. Do not include any introductory explanation, definitions, structural outlines, or closing remarks. Output only the raw poem verses from the very first word to the final line.`
>
> **Output:**  
> *Moonlit waves caress the sandy shore (A)*  
> *Gentle whispers of a love forever more (A)*  
> *Echoes of a memory that time cannot bore (A)*  
> *A bittersweet reminder of what we adore (A)*  
> *And in its beauty, a heart does explore (A)*  
> *The stars above, a twinkling show (B)*  
> *Their celestial dance, a wondrous glow (B)*  
> *The world is hushed, in quiet, darkened low (B)*  
> *The night's dark veil, a mystery does grow (B)*  
> *In darkness, fears and doubts do quietly go (B)*  
> *Moonlit waves caress the sandy shore (A)*  
> *Gentle whispers of a love forever more (A)*  
> *Echoes of a memory that time cannot bore (A)*  
> *A bittersweet reminder of what we adore (A)*  
> *And in its beauty, a heart does explore (A)*  
> *The stars above, a twinkling show (B)*  
> *Their celestial dance, a wondrous glow (B)*  
> *The world is hushed, in quiet, darkened low (B)*  
> *The night's dark veil, a mystery does grow (B)*  
> *In darkness, fears and doubts do quietly go (B)*  
> *And in the silence, a peaceful heart does stay (C)*  
> *In stillness, a calm and quiet soul does sway (C)*  
> *In darkness, a new dawn does slowly break (C)*  
> *And in its light, a brighter day does make (D)*  
> *And in its warmth, a heart does start to awake (D)*  
> *And in its hope, a soul does start to partake (D)*  
> *Moonlit waves caress the sandy shore so free (E)*  
> *And in its peace, a heart does start to see (E)*  
> *And in its calm, a soul does start to be (E)*  
> *And in the stillness, a heart does start to breathe (C)*  
> *And in the quiet, a soul does start to grieve (C)*  
> *And in the darkness, a heart does start to flee (F)*  
> *(Natural EOS completion at 409 tokens)*

</details>

---

### 2. Hysteresis Parameter Sensitivity & Pareto Bounds
Empirical 9-point grid sweep mapping the threshold space $(\theta_{\text{low}}, \theta_{\text{high}})$ against sustained throughput and speculative drafting ratio on the RTX 5090 Blackwell rig:

<p align="center">
  <img src="./assets/sdsie_parameter_sweep_pareto.png" alt="SDSIE Parameter Sweep Pareto Bounds" width="100%">
</p>

| Operational Control Mode | $\theta_{\text{low}}$ (bits) | $\theta_{\text{high}}$ (bits) | Speculation Ratio | Primary Workload Profile |
| :--- | :---: | :---: | :---: | :--- |
| **🛡️ Conservative Fortress** | `0.35` | `1.00` – `1.50` | **0.0%** | Formal logic, legal synthesis, mathematical proofs (zero draft-rejection tolerance). |
| **⚡ Balanced Baseline** | `0.55` | `1.25` | **5.9%** | General chat, structured poetry, multi-turn prose (default production profile). |
| **🚀 High-Throughput Burst** | `0.75` | `1.75` | **33.8%** | Conversational filler, deterministic templating, batch summarization. |

---

### 3. Comprehensive Hardware Telemetry Matrix
<p align="center">
  <img src="./assets/sdsie_telemetry_matrix.png" alt="SDSIE Hardware Telemetry Matrix" width="100%">
</p>

---

## 🚀 Quickstart & Serving Stack

### 1. Installation
```bash
git clone https://github.com/Creepybits/software-defined-stochastic-inference-engine.git
cd software-defined-stochastic-inference-engine
pip install -e .
```

### 2. Launch the Inference Server & Web UI
```bash
# Terminal 1: Launch the OpenAI-compatible SDSIE inference engine
python sdsie_server.py --port 8000 --model meta-llama/Llama-3.1-8B-Instruct

# Terminal 2: Launch the local dependency-free Web UI (port 5000)
python web_ui.py
```
Open your browser at ```http://localhost:5000```

### 3. Micro-Verification Test Suite
```bash
# 1. Micro-Kernel Sanity Check (28.6 µs isolated GEMM)
python test_vllm_sdsie.py

# 2. Full Llama-3.1-8B SwiGLU MLP Block Forward Pass (155.8 µs)
python test_transformer_block.py

# 3. Entropy Clutch Verification
python test_spec_controller.py

# 4. Run the 9-Point Parameter Sweep & Pareto Plotter
python sweep_real_model.py
python plot_parameter_sweep.py
```
___
## 🌐 Ecosystem Integration & Upstream RFCs
SDSIE is actively engaging the open-source inference ecosystem to upstream dynamic entropy-gated speculative scheduling:
* **vLLM (Cloud Enterprise)**: [Feature RFC #54082](https://github.com/vllm-project/vllm/issues/54082)
* **SGLang (Fast Serving Runtime)**: [Feature RFC #36715](https://github.com/sgl-project/sglang/issues/36715)
* **llama.cpp (Local Edge & Ollama Backend)**: [Research Issue #27821](https://github.com/ggml-org/llama.cpp/issues/27821)

___
## 📜 Scientific Archival & Citation
SDSIE is permanently archived under CERN / Zenodo:
* Master Concept DOI: [10.5281/zenodo.21499379](https://doi.org/10.5281/zenodo.21499379)
* Latest Formal Paper Release (v1.2.2): [10.5281/zenodo.22129912](https://zenodo.org/records/22129912)

```bibtex
@article{jacklin2026sdsie,
  title={Software-Defined Stochastic Inference Engine (SDSIE): Energy-Proportional LLM Serving via Fused SRAM Dequantization and Entropy-Gated Speculative Control},
  author={Jacklin, Zanno},
  journal={IEEE Transactions on Sustainable Computing / ACM MLSys Baseline},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.22129912}
}
```
___
## 👨‍💻 Principal Investigator  
Zanno Jacklin  
Principal Investigator, Creepybits (Borås, Sweden)  
Email: business@zanno.se | ORCID: [0000-0001-9164-4650](https://orcid.org/0000-0001-9164-4650)  
Website: [zanno.se](https://zanno.se/) | Project Portal: [sdsie.github.io](https://sdsie.github.io/)  

Licensed under the [Apache-2.0 License](LICENSE).

