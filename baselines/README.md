# Baselines (cloned)

This folder contains SOTA baseline repositories cloned to support the Unified Conversion Framework (UCF) project.

Cloned repos (shallow clones):

- `distilling-step-by-step` — https://github.com/google-research/distilling-step-by-step
- `llm-awq` — https://github.com/mit-han-lab/llm-awq
- `streaming-llm` — https://github.com/mit-han-lab/streaming-llm
- `qlora` — https://github.com/artidoro/qlora
- `MobileLLM` — https://github.com/facebookresearch/MobileLLM

Notes:
- `GEEP` (parameter-efficient debiasing) repository link was not provided. Please add the repo URL or a local copy and I will integrate it.
- These are shallow clones (`--depth 1`) to reduce download size. If you need full history, reclone without `--depth`.

Next steps:
1. Create per-repo wrapper scripts in `D:\ucf_framework\baselines` to standardize training/inference commands.
2. Produce a unified `ucf` conda environment and per-repo `requirements.txt` files.
3. Implement evaluation harness at `D:\ucf_framework\src` that can call each baseline and collect metrics.

If you want, I can now scaffold the wrapper scripts and a unified evaluation harness.
