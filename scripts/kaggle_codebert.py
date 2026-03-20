"""Kaggle notebook preamble for CodeBERT polarity experiment.

Copy this entire file into a Kaggle notebook cell (Python, GPU T4 x2 enabled).
It clones the repo, installs deps, and runs the experiment.

Expected runtime: ~2-3 hours on T4 GPU.
Output: v31_codebert_polarity.json (download from output/tables/)
"""

# %% [markdown]
# # CodeBERT Polarity Analysis for ATLAS
# Tests whether code-specialized encoders (CodeBERT 125M, 768-dim) break
# the polarity reversal pattern observed with MiniLM (22M, 384-dim).

# %% Cell 1: Setup
import subprocess
import os

# Clone repo
if not os.path.exists("research-workspace"):
    subprocess.run([
        "git", "clone",
        "https://github.com/shivamakhauri04/research-workspace.git",
    ], check=True)

os.chdir("research-workspace/experiments/trustlens-auditable-agent-trust")

# Install dependencies
subprocess.run([
    "pip", "install", "-q",
    "transformers>=4.35.0",
    "torch",
    "sentence-transformers>=2.2.0",
    "orjson",
    "scipy",
    "scikit-learn",
    "datasets",  # for HuggingFace dataset loading
    "accelerate",
], check=True)

# %% Cell 2: Run experiment
import sys
sys.path.insert(0, "../../lib")
sys.path.insert(0, "src")
os.environ["PYTHONUNBUFFERED"] = "1"

# Run with both CodeBERT and CodeT5+
subprocess.run([
    sys.executable, "scripts/run_codebert_polarity.py",
    "--kaggle",
    "--models", "codebert", "codet5p",
    "--max-traces", "5000",
], check=True)

# %% Cell 3: Display results
import json
from pathlib import Path

result_path = Path("output/tables/v31_codebert_polarity.json")
if result_path.exists():
    results = json.loads(result_path.read_text())
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    for key, res in results["results"].items():
        pol = res["polarity"]
        proto = res.get("prototype_scorer", {})
        print(f"\n  {res['model_name']} ({res['model_params']}, {res.get('embedding_dim', '?')}-dim):")
        print(f"    Within cos:  {pol['within_cos_mean']:+.3f} (neg: {pol['within_neg_pct']:.1f}%)")
        print(f"    Between cos: {pol['between_cos_mean']:+.3f} (neg: {pol['between_neg_pct']:.1f}%)")
        if proto:
            print(f"    Proto AUROC: W={proto.get('within_auroc_mean', 0):.3f}, "
                  f"B={proto.get('between_auroc_mean', 0):.3f}")
            print(f"    Anti-predict: {proto.get('anti_predict_pct', 0):.1f}%")

    # Copy to Kaggle output for download
    import shutil
    kaggle_out = Path("/kaggle/working/v31_codebert_polarity.json")
    shutil.copy(result_path, kaggle_out)
    print(f"\n  Results copied to {kaggle_out}")
else:
    print("ERROR: Results file not found!")
