"""
Scan `baselines/` for common requirement files (requirements.txt, pyproject.toml, environment.yml)
and produce:
 - `requirements_ucf.txt` : aggregated pip requirements (deduped)
 - `environment_ucf.yml` : conda env additions (manual review suggested)

This is a helper to make a unified conda environment for the UCF workspace.
"""
import os
import re
from glob import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BASELINES = os.path.join(ROOT, "baselines")

reqs = set()
conda_deps = set()


def parse_requirements_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            reqs.add(line)


def parse_pyproject(path):
    # naive parse to find dependency table under [project] or poetry
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # find common dependencies like transformers, datasets, accelerate
    for pkg in ["transformers", "datasets", "accelerate", "peft", "bitsandbytes", "sentencepiece"]:
        if re.search(pkg, text, re.IGNORECASE):
            reqs.add(pkg)


def parse_env_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                pkg = line[2:].strip()
                # quick heuristic: if contains '=' treat as conda dep
                if "=" in pkg:
                    conda_deps.add(pkg)
                else:
                    reqs.add(pkg)


for pattern in (os.path.join(BASELINES, "**", "requirements.txt"), os.path.join(BASELINES, "**", "pyproject.toml"), os.path.join(BASELINES, "**", "environment.yml")):
    for p in glob(pattern, recursive=True):
        try:
            if p.endswith("requirements.txt"):
                parse_requirements_txt(p)
            elif p.endswith("pyproject.toml"):
                parse_pyproject(p)
            elif p.endswith("environment.yml") or p.endswith("environment.yaml"):
                parse_env_yaml(p)
        except Exception:
            pass

# write outputs
out_reqs = os.path.join(ROOT, "requirements_ucf.txt")
with open(out_reqs, "w", encoding="utf-8") as f:
    for r in sorted(reqs):
        f.write(r + "\n")

out_env = os.path.join(ROOT, "environment_ucf.yml")
with open(out_env, "w", encoding="utf-8") as f:
    f.write("name: ucf_extended\nchannels:\n  - defaults\n  - conda-forge\ndependencies:\n")
    for d in sorted(conda_deps):
        f.write(f"  - {d}\n")
    f.write("  - pip\n  - pip:\n")
    for r in sorted(reqs):
        f.write(f"    - {r}\n")

print("Wrote:", out_reqs, "and", out_env)
