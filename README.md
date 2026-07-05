# XAI Mini Project – AIFB Dataset, Strategy 1

**Course:** Explainable Artificial Intelligence – Dr. Stefan Heindorf, Paderborn University  
**Dataset:** AIFB (persons & publications → research group classification)  
**Strategy:** Strategy 1 – Explainability methods for GNNs (GNNExplainer)

---

## Team Members

| Name                    | IMT Account | Matriculation Number |
|------------------------ |-------------|----------------------|
|  Anmol Raj Srivastav    |             |     4064093          |
|                         |             |                      |


---

## Project Overview

This project trains a Graph Convolutional Network (GCN) on the AIFB knowledge
graph and generates explanations using GNNExplainer from PyTorch Geometric.

**Task:** Given a person node in the AIFB knowledge graph, predict which of the
4 research groups they belong to:
- Business Information & Communication Systems
- Efficient Algorithms
- Knowledge Management
- Complexity Management

---

## Repository Structure

```
.
├── run_all.py               # One-command runner (installs packages + runs everything)
├── requirements.txt         # Python dependencies
├── README.md
├── scripts/
│   ├── setup.py             # Download AIFB dataset
│   ├── build_mapping.py     # Deterministic dataset build + label mappings
│   ├── data_analysis.py     # RDF graph statistics & visualisations
│   ├── model_training.py    # R-GCN training & evaluation
│   ├── explanations.py      # GNNExplainer with subgraph visualisations
│   ├── evaluation.py        # Fidelity+, Fidelity-, Sparsity, Stability metrics
│   └── aifb_utils.py        # Shared dataset helpers
├── data/                    # AIFB dataset (downloaded by setup.py)
├── models/                  # Trained model & explanation checkpoints (.pt)
├── plots/                   # All generated figures (.png)
├── results/                 # Metrics CSVs + generated explanations report
└── report/                  # Written project report (LaTeX)
```

> **Note:** All scripts must be run **from the repository root** (e.g.
> `python scripts/setup.py`), so that relative paths like `./data` resolve
> correctly. `run_all.py` does this automatically.

---

## Requirements

- Python 3.12 or 3.13
- macOS / Linux / Windows
- No GPU required (CPU is fine)

---

## Quick Start (one command)

Instead of following the manual steps below, you can run the **entire project
with a single command**. It installs all packages and executes every script in
the correct order (no plot windows to close – all figures are saved as PNGs):

```bash
python run_all.py
```

Optionally, but recommended, create a virtual environment first:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python run_all.py
```

If packages are already installed, skip installation with:

```bash
python run_all.py --no-install
```

The manual step-by-step instructions below are equivalent.

---

## Setup Instructions

### Step 1 – Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/xai-project.git
cd xai-project
```

### Step 2 – Create and activate virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal line.

### Step 3 – Install dependencies

Run each line one at a time and wait for each to finish:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install torch-geometric
```

```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

```bash
pip install rdflib pandas matplotlib scikit-learn networkx ipykernel
```

---

## Reproducing Results

Run scripts **in this exact order**. Make sure `(venv)` is active in your
terminal before running any script.

### Step 1 – Download dataset
```bash
python3 scripts/setup.py
```
Downloads the AIFB dataset into `./data/`.

Expected output:
```
✅ Dataset already downloaded.
✅ Dataset already extracted.
```

---

### Step 2 – Data analysis
```bash
python3 scripts/data_analysis.py
```
Loads the RDF graph, prints statistics, saves two plots.

Expected output:
```
Loading RDF graph ...
✅ Loaded 29,226 triples.

── Basic Statistics ──────────────────────────────────
  Total triples                        29,226
  Unique subjects                       2,506
  Unique predicates                        53
  Unique objects                        6,654

── Class Distribution ────────────────────────────────
  Business Info & Communication Systems   73
  Knowledge Management                    60
  Efficient Algorithms                    28
  Complexity Management                   16
```

Output files: `class_distribution.png`, `top_predicates.png`

---

### Step 3 – Train model
```bash
python3 scripts/model_training.py
```
Trains an R-GCN for 50 epochs. When the plot window appears, **close it**
so the script can finish saving the model.

Expected output:
```
✅ Dataset loaded: AIFB
   Nodes          : 8,285
   Edges          : 58,086
   Relation types : 90
   Classes        : 4
   Train nodes    : 140
   Test nodes     : 36

── Training ──────────────────────────────────────────
  Epoch  10 | Loss 0.0733 | Train Acc 0.9786 | Test Acc 0.8611
  Epoch  20 | Loss 0.0262 | Train Acc 1.0000 | Test Acc 0.9444
  Epoch  30 | Loss 0.0052 | Train Acc 1.0000 | Test Acc 0.9722
  Epoch  40 | Loss 0.0021 | Train Acc 1.0000 | Test Acc 0.9722
  Epoch  50 | Loss 0.0034 | Train Acc 1.0000 | Test Acc 0.9722

── Final Evaluation ──────────────────────────────────
  Train  Accuracy : 1.0000   Macro-F1 : 1.0000
  Test   Accuracy : 0.9722   Macro-F1 : 0.9699

✅ Model saved to ./model_rgcn.pt
✅ Data saved to ./data_object.pt
```

Output files: `training_curves.png`, `model_rgcn.pt`, `data_object.pt`

---

### Step 4 – Generate explanations
```bash
python3 scripts/explanations.py
```
Trains a GCN, runs GNNExplainer on 5 correctly classified test nodes,
saves one subgraph visualisation per node. Close each plot window to continue.

Expected output:
```
✅ Data loaded: 8285 nodes, 58086 edges
── Training GCN for explanation ──────────────────────
  Epoch  50 | Loss 0.1992 | Train Acc 1.0000 | Test Acc 0.9167
✅ 33 correctly classified test nodes found.
── GNNExplainer ──────────────────────────────────────
  Node  6540 | pred=0 (Business Info & Communication Systems) | important edges: 26
  Node   500 | pred=1 (Efficient Algorithms) | important edges: 54
  Node  4531 | pred=3 (Complexity Management) | important edges: 11
  Node  6958 | pred=1 (Efficient Algorithms) | important edges: 56
  Node  4570 | pred=0 (Business Info & Communication Systems) | important edges: 5
✅ All explanations saved to ./explanations.pt
```

Output files: `explanation_node_*.png` (5 files), `explanations.pt`

---

### Step 5 – Evaluate explanations
```bash
python3 scripts/evaluation.py
```
Computes Fidelity+, Fidelity−, Sparsity and Stability across 20 test nodes.
Takes ~5 minutes. Close the plot window at the end to finish.

Expected output:
```
── Evaluating 20 nodes ───────────────────────────────
  Node  6540 | F+=1.00 F-=1.00 Spars=1.00 Stab=0.99
  ...

── Aggregated Metrics ────────────────────────────────
  fidelity+   : 0.8500 ± 0.3600
  fidelity-   : 0.9000 ± 0.3000
  sparsity    : 0.9995 ± 0.0010
  stability   : 0.9800 ± 0.0200
```

Output files: `eval_metrics.png`, `explanation_evaluation.csv`

---

## Output Files Summary

| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart of research group sizes |
| `top_predicates.png` | Most frequent edge types in RDF graph |
| `training_curves.png` | Loss and accuracy over training epochs |
| `model_rgcn.pt` | Saved R-GCN model weights |
| `explanation_node_*.png` | Subgraph explanation for each node |
| `eval_metrics.png` | Bar chart of explanation quality metrics |
| `explanation_evaluation.csv` | Per-node metric scores |

---

## Explanation Metrics

| Metric | Description | Higher is better? |
|--------|-------------|-------------------|
| Fidelity+ | Prediction changes when important edges removed | ✅ Yes |
| Fidelity- | Prediction preserved with only important edges | ✅ Yes |
| Sparsity | Fraction of edges not selected as important | ✅ Yes |
| Stability | Consistency of masks across repeated runs | ✅ Yes |

---

## Common Errors

**`(venv)` not showing in terminal**
```bash
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

**`ModuleNotFoundError: No module named 'torch'`**
Your venv is not active. Run the activate command above first.

**`FileNotFoundError: model_rgcn.pt`**
Run `model_training.py` first and make sure to close the plot window
so the script saves the model file.

**Plot windows block the script**
Always close plot windows by clicking the X button so the script continues.

---

## References

1. Schlichtkrull et al. (2018). *Modeling Relational Data with Graph Convolutional Networks.*
2. Ying et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.*
3. Fey & Lenssen (2019). *Fast Graph Representation Learning with PyTorch Geometric.*
4. Ristoski & Paulheim (2016). *RDF2Vec: RDF Graph Embeddings and Their Applications.*