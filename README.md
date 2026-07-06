# XAI Mini Project – AIFB Dataset, Strategy 1

**Course:** Explainable Artificial Intelligence – Dr. Stefan Heindorf, Paderborn University  
**Dataset:** AIFB (persons & publications → research group classification)  
**Strategy:** Strategy 1 – Explainability methods for GNNs (GNNExplainer)

---

## Team Members

| Name                    | Matriculation Number |
|------------------------ |----------------------|
|  Anmol Raj Srivastav    |     4064093          |
|  Janhavi Palav          |     4063279          |


---

## Project Overview

This project trains a Relational Graph Convolutional Network (FastRGCN) on the
AIFB knowledge graph and generates explanations using GNNExplainer from
PyTorch Geometric.

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
├── results/                 # Metrics CSVs with explanation results
└── Report_Mini_Project.pdf  # Written project report (PDF)
```

> **Note:**  Run `run_all.py` and it will run all the scripts accordingly and automatically.

---

## Requirements

- Python 3.12 or 3.13
- macOS / Linux / Windows
- No GPU required (CPU is fine)

---

## Quick Start (one command execution)

Instead of following the manual steps below, you can run the **entire project
with a single command**. It installs all packages and executes every script in
the correct order (no plot windows to close – all figures are saved as PNGs):

**1. Clone the repository:**

```bash
git clone git@github.com:anmolraj-hub/XAI_Mini_Project.git
cd XAI_Mini_Project
```
**2. create a virtual environment first and run the run_all.py file**


```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python run_all.py
```

If packages are already installed, skip installation with:

```bash
python run_all.py --no-install
```

The full pipeline takes roughly **25–30 minutes** on a laptop CPU (training
~2 min, explanations ~10 min, evaluation ~15 min).


---

All figures are saved to `plots/`.
---

## References

1. Schlichtkrull et al. (2018). *Modeling Relational Data with Graph Convolutional Networks.*
2. Ying et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.*
3. Fey & Lenssen (2019). *Fast Graph Representation Learning with PyTorch Geometric.*
4. Ristoski & Paulheim (2016). *RDF2Vec: RDF Graph Embeddings and Their Applications.*