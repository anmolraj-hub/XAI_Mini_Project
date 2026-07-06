"""
XAI Mini Project - One-Command Runner
AIFB Dataset | Strategy 1: GNN Explainability

Usage
-----
    python run_all.py                # installs all packages, then runs the full pipeline
    python run_all.py --no-install   # skip package installation, just run the pipeline


"""

import os
import subprocess
import sys
import time


# Configuration


# pip install commands, run in this order..
INSTALL_COMMANDS = [
    [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
    [sys.executable, "-m", "pip", "install",
     "torch", "torchvision", "torchaudio",
     "--index-url", "https://download.pytorch.org/whl/cpu"],
    [sys.executable, "-m", "pip", "install", "torch-geometric"],
    [sys.executable, "-m", "pip", "install",
     "rdflib", "pandas", "matplotlib", "scikit-learn", "networkx"],
]

# Optional PyG extensions. These sometimes fail to build on certain
# Python versions; the project still runs without them, so failures
# here are reported but NOT fatal.
OPTIONAL_INSTALL_COMMANDS = [
    [sys.executable, "-m", "pip", "install",
     "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
     "-f", "https://data.pyg.org/whl/torch-2.3.0+cpu.html"],
]

# Pipeline scripts (inside scripts/), in the exact order they must run.
PIPELINE = [
    ("setup.py",          "Download AIFB dataset"),
    ("build_mapping.py",  "Build deterministic dataset + label mappings"),
    ("data_analysis.py",  "RDF graph statistics & visualisations"),
    ("model_training.py", "Train & evaluate R-GCN"),
    ("explanations.py",   "GNNExplainer subgraph explanations"),
    ("evaluation.py",     "Fidelity+, Fidelity-, Sparsity, Stability metrics"),
]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
OUTPUT_DIRS = ["data", "models", "plots", "results"]



# Helpers


def banner(text: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}", flush=True)


def run_command(cmd, fatal: bool = True) -> bool:
    """Run a shell command, streaming its output. Return True on success."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        if fatal:
            print(f"\n[ERROR] Command failed with exit code {result.returncode}: "
                  f"{' '.join(cmd)}")
            sys.exit(result.returncode)
        else:
            print("[WARN] Optional step failed - continuing anyway.")
            return False
    return True


def install_packages() -> None:
    banner("STEP 0: Installing packages (this may take a few minutes)")
    for cmd in INSTALL_COMMANDS:
        run_command(cmd, fatal=True)
    for cmd in OPTIONAL_INSTALL_COMMANDS:
        run_command(cmd, fatal=False)
    print("\n[OK] All packages installed.")


def run_pipeline() -> None:
    # Ensure all output folders exist.
    for folder in OUTPUT_DIRS:
        os.makedirs(os.path.join(PROJECT_DIR, folder), exist_ok=True)

    # Reproducibility + no blocking plot windows:
    #  - MPLBACKEND=Agg    -> plt.show() never blocks; figures still saved
    #  - PYTHONHASHSEED=0  -> Python set/dict iteration order is fixed
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONHASHSEED"] = "0"

    total_start = time.time()

    for i, (script, description) in enumerate(PIPELINE, start=1):
        script_path = os.path.join(SCRIPTS_DIR, script)
        if not os.path.exists(script_path):
            print(f"[ERROR] {script} not found in {SCRIPTS_DIR}. "
                  f"Make sure the repository structure is intact.")
            sys.exit(1)

        banner(f"STEP {i}/{len(PIPELINE)}: scripts/{script}  -  {description}")
        start = time.time()
        # cwd=PROJECT_DIR so the scripts' relative paths
        # (./data, ./models, plots/, results/) resolve from the repo root.
        result = subprocess.run([sys.executable, script_path],
                                cwd=PROJECT_DIR, env=env)
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"\n[ERROR] {script} failed after {elapsed:.1f}s "
                  f"(exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

        print(f"\n[OK] {script} finished in {elapsed:.1f}s.")

    total = time.time() - total_start
    banner(f"ALL DONE - full pipeline completed in {total/60:.1f} minutes")
    print("Generated outputs:")
    print("  plots/    class_distribution.png, top_predicates.png,")
    print("            training_curves.png, global_explanation.png,")
    print("            local_exp_*.png, eval_metrics.png")
    print("  models/   model_rgcn.pt, explanations.pt")
    print("  results/  person_explanations_report.md,")
    print("            person_explanations_summary.csv, explanation_evaluation.csv")



# Main


if __name__ == "__main__":
    skip_install = "--no-install" in sys.argv

    print("XAI Mini Project - AIFB Dataset - Full Pipeline Runner")
    print(f"Python : {sys.version.split()[0]} ({sys.executable})")
    print(f"Folder : {PROJECT_DIR}")

    if skip_install:
        print("\n(--no-install given: skipping package installation)")
    else:
        install_packages()

    run_pipeline()
