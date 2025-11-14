# Unified Conversion Framework (UCF) — Research & Implementation Plan

**Project Title:** Unified Conversion Framework for Efficient, Coherent, Fair, Conversational AI on Edge Devices

**Workstation (target dev / eval machine):**
- Model: HP Z4 Tower G5 Business Workstation
- CPU: Intel Xeon W5-2565X (18 cores)
- Chipset: Intel W790
- RAM: 64 GB DDR5 4800 MHz
- GPU: NVIDIA RTX 2000 ADA 16 GB (4 mini-DisplayPort)

This file summarizes strategy, integration plan, experiments and recommended publication venues.

**High-level Research Goals:**
- Build a modular unified workspace that integrates distillation, quantization, PEFT, long-context streaming, and mobile optimization techniques.
- Produce a flexible evaluation harness to run controlled ablation studies across: accuracy/utility, coherence, fairness (gender bias & group fairness), and resource-efficiency (memory, latency, energy).
- Demonstrate state-of-the-art tradeoffs on edge device constraints (16 GB GPU, limited CPU/RAM), and provide reproducible scripts and best-practices.

**Baseline repos integrated:**
- Distillation: `distilling-step-by-step` (ACL 2023)
- Quantization: `llm-awq` (MLSys 2024)
- Long-context / streaming: `streaming-llm` (EMNLP/ICLR papers)
- PEFT & quantized finetuning: `qlora` (NeurIPS 2023)
- Mobile & efficient LLM runtime ideas: `MobileLLM`
- (Pending) GEEP: parameter-efficient debiasing — need repo link or local copy

**Core implementation strategy (practical steps):**
1. Environment & platform
   - Use Miniconda on Windows. Create env from `D:\ucf_framework\environment.yml` and follow the PyTorch installation notes for GPU support.
   - Prefer Python 3.10 for broad library support.

2. Reproducible workspace layout (scaffolded):
   - `D:\ucf_framework\baselines/` — baseline repos (already cloned).
   - `D:\ucf_framework\src/ucf/` — common wrappers, orchestration scripts, evaluation harness.
   - `D:\ucf_framework\data/` — datasets for experiments and evaluation.
   - `D:\ucf_framework\results/` — standardized outputs for metrics, logs, and checkpoints.

3. Integration plan (for each baseline)
   - Add a uniform CLI wrapper `run_{baseline}.py` in `src/ucf/wrappers/` that: normalizes arguments, sets seeds, logs metrics, and writes outputs in a common format (JSONL + CSV for metrics).
   - For distillation: replicate core training recipe; implement an option to load smaller student models and distillation targets.
   - For AWQ & QLoRA: add quantization + finetuning pipelines with toggles (4-bit, 8-bit, AWQ per-layer settings).
   - For StreamingLLM: integrate streaming inference tests and measure latency/throughput on long inputs.
   - For MobileLLM: import runtime optimizations and test on the target GPU and CPU-only scenarios.

4. Evaluation & ablation
   - Standard metrics: accuracy (task-specific), perplexity, ROUGE/BLEU (as applicable), and retrieval/QA metrics.
   - Coherence metrics: automated coherence scoring (e.g., ADVISOR metrics or model-based coherence scorers), human eval plan for final paper.
   - Fairness metrics: use GEEP methodology for gender fairness (once repo available); measure demographic parity, equalized odds, and per-group performance gaps.
   - Efficiency metrics: GPU memory usage (peak), latency (p50/p90), throughput, and energy (if possible).

5. Experimental design and ablation axes
   - Distillation (teacher size / dataset size / instruction tuning) vs direct PEFT finetuning (QLoRA).
   - Quantization: post-training AWQ vs QLoRA 4-bit finetuning (compare performance vs memory/time tradeoffs).
   - Streaming vs full-context inference on long inputs: quality vs latency tradeoff.
   - Debiasing (GEEP) vs baseline: measure fairness vs accuracy tradeoffs.

6. Reproducibility
   - Save experiment config (YAML/JSON) for each run.
   - Attach exact commit hashes for each baseline repo and provide seeded random state.
   - Provide small-scale checkpoints (and instructions to reproduce larger runs locally or on cloud).

**Hardware & runtime suggestions (given RTX 2000 ADA 16GB):**
- For large LLMs, target 4-bit / AWQ 8-bit where possible to run on 16GB.
- Use QLoRA-style offloading and gradient checkpointing when training/finetuning.
- Batch sizes will be small; prefer gradient accumulation to approximate larger batches.
- For quantization experiments, test post-training quantization and quantized fine-tuning separately.

**Publication targets & recommended timeline (high-level)**
- Top NLP / ML venues suitable for this work:
  - ACL / EMNLP / NAACL (NLP-focused; strong fit for distillation + fairness + conversational QA)
  - NeurIPS / ICML / ICLR (ML-focused; strong fit for methodology / quantization + PEFT innovations)
  - MLSys (systems & on-device inference — strong fit for AWQ / MobileLLM results)

- Typical submission windows (check official CFP pages for exact 2026 deadlines):
  - NeurIPS: submissions typically in May–June (paper + reproducibility checks).
  - ICLR: submissions typically in September–October.
  - ICML: submissions often January–February.
  - ACL / EMNLP / NAACL: usually have late-year (Nov–Jan) or mid-year calls—check CFPs.
  - MLSys: usually early-to-mid year (Feb–Apr) submissions.

Recommended publishing strategy:
- Aim for one major systems/methods paper (NeurIPS/ICLR/ICML) showing unified methods (distillation+PEFT+quantization) and learning contributions.
- Submit a second, focused systems paper to MLSys showing on-device performance and engineering contributions.
- For strong NLP evaluation/analysis (coherence + fairness + conversational quality), target ACL or EMNLP.

**Practical timeline (example, adapt to exact CFP dates):**
- Nov–Dec 2025: Finish integration, cloning (done), scaffolding wrappers, environment install.
- Jan–Mar 2026: Implement baseline pipelines (distill, AWQ, QLoRA, streaming), initial small-scale experiments.
- Apr–Jun 2026: Large-scale ablation experiments and human eval planning.
- Jun–Aug 2026: Paper writing and reproducibility package assembly.
- Sept–Nov 2026: Submit to target conferences depending on CFP.

**Deliverables I'll help implement next (suggested immediate steps):**
- Scaffold `src/ucf/wrappers/` with template runners for each baseline.
- Implement `src/ucf/eval_harness.py` that standardizes metrics and outputs.
- Create `D:\ucf_framework\experiments\` skeleton and a sample experiment config + run script.

---

If you want, I can now:
- scaffold the wrappers and evaluation harness in `D:\ucf_framework\src` (I will create the files and a small smoke test), or
- set up the conda env skeleton and provide exact install commands for Windows CUDA/PyTorch matching your machine.

Which of these two should I do next? If you want both, I'll scaffold wrappers first and then prepare the environment install notes and environment activation commands.
