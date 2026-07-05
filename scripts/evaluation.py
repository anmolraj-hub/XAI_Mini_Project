"""
XAI Mini Project - Step 5: Explanation Evaluation
AIFB Dataset | Strategy 1: GNN Explainability

This file evaluates explanations for the SAME FastRGCN model that is trained in
model_training.py and explained in explanations.py.

Metrics:
  1. Fidelity+  - prediction changes when important edges are REMOVED
  2. Fidelity-  - prediction stays when only important edges are KEPT
  3. Sparsity   - fraction of edges NOT selected as important
  4. Stability  - consistency of explanations across repeated runs
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "./.matplotlib_cache")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import FastRGCNConv


# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "./data/pyg_aifb"
MODEL_PATH = "./models/model_rgcn.pt"
HIDDEN_DIM = 16
SEED = 42
EXPLAINER_EPOCHS = 100
EDGE_IMPORTANCE_THRESHOLD = 0.5

torch.manual_seed(SEED)
torch.set_num_threads(1)                                  # reproducible CPU runs
torch.use_deterministic_algorithms(True, warn_only=True)  # reproducible CPU runs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model definitions: must match model_training.py and explanations.py ───────
class FastRGCN(torch.nn.Module):
    def __init__(self, num_nodes, emb_dim, hidden_channels,
                 out_channels, num_relations):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, emb_dim)
        self.conv1 = FastRGCNConv(emb_dim, hidden_channels, num_relations)
        self.conv2 = FastRGCNConv(hidden_channels, out_channels,
                                  num_relations)

    def forward(self, x, edge_index, edge_type=None):
        x = self.embedding(x)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index, edge_type)


class FastRGCNExplainable(torch.nn.Module):
    """GNNExplainer wrapper that receives pre-computed node embeddings."""

    def __init__(self, conv1, conv2):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, x, edge_index, edge_type=None):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index, edge_type)


# ── Loading ───────────────────────────────────────────────────────────────────
def load_model_and_data():
    dataset = Entities(root=DATA_ROOT, name="AIFB")
    data = dataset[0].to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device,
                            weights_only=False)
    num_classes = checkpoint["num_classes"]
    num_relations = checkpoint["num_relations"]
    num_nodes = checkpoint["num_nodes"]
    emb_dim = checkpoint["emb_dim"]

    full_model = FastRGCN(
        num_nodes=num_nodes,
        emb_dim=emb_dim,
        hidden_channels=HIDDEN_DIM,
        out_channels=num_classes,
        num_relations=num_relations,
    ).to(device)
    full_model.load_state_dict(checkpoint["model_state"])
    full_model.eval()

    node_ids = torch.arange(data.num_nodes, device=device)
    with torch.no_grad():
        x = full_model.embedding(node_ids)

    model = FastRGCNExplainable(full_model.conv1, full_model.conv2).to(device)
    model.eval()

    print("✅ Loaded saved FastRGCN model for explanation evaluation.")
    print(f"   Nodes          : {data.num_nodes:,}")
    print(f"   Edges          : {data.edge_index.shape[1]:,}")
    print(f"   Relation types : {num_relations}")
    print()
    return model, data, x


# ── Explainer and node selection ──────────────────────────────────────────────
def make_explainer(model, epochs=EXPLAINER_EPOCHS):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )


@torch.no_grad()
def select_correct_test_nodes(model, data, x, n=20):
    """Selects correctly classified test nodes from the saved FastRGCN."""
    model.eval()
    logits = model(x, data.edge_index, data.edge_type)
    preds = logits.argmax(dim=1)

    selected = []
    for offset, node_tensor in enumerate(data.test_idx):
        node_idx = int(node_tensor.item())
        if preds[node_idx].item() == data.test_y[offset].item():
            selected.append(node_idx)
        if len(selected) == n:
            break
    return selected


# ── Metrics ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_pred(model, x, edge_index, edge_type, node_idx):
    model.eval()
    out = model(x, edge_index, edge_type)
    return out[node_idx].argmax().item()


def fidelity_plus(model, data, x, node_idx, edge_mask,
                  threshold=EDGE_IMPORTANCE_THRESHOLD):
    """Good if prediction changes when important edges are removed."""
    full_pred = get_pred(
        model, x, data.edge_index, data.edge_type, node_idx)

    keep_mask = ~(edge_mask > threshold)
    if keep_mask.sum() == 0:
        return 0.0

    reduced_edge_index = data.edge_index[:, keep_mask]
    reduced_edge_type = data.edge_type[keep_mask]
    reduced_pred = get_pred(
        model, x, reduced_edge_index, reduced_edge_type, node_idx)
    return float(full_pred != reduced_pred)


def fidelity_minus(model, data, x, node_idx, edge_mask,
                   threshold=EDGE_IMPORTANCE_THRESHOLD):
    """Good if prediction stays the same using only important edges."""
    full_pred = get_pred(
        model, x, data.edge_index, data.edge_type, node_idx)

    keep_mask = edge_mask > threshold
    if keep_mask.sum() == 0:
        return 0.0

    kept_edge_index = data.edge_index[:, keep_mask]
    kept_edge_type = data.edge_type[keep_mask]
    kept_pred = get_pred(
        model, x, kept_edge_index, kept_edge_type, node_idx)
    return float(full_pred == kept_pred)


def sparsity(edge_mask, threshold=EDGE_IMPORTANCE_THRESHOLD):
    """Fraction of edges not selected as important. Higher is more compact."""
    important_edges = (edge_mask > threshold).sum().item()
    return 1.0 - (important_edges / len(edge_mask))


def stability(explainer, data, x, node_idx, runs=1, base_mask=None):
    """Cosine similarity between repeated edge masks. Higher is more stable.

    If base_mask is given, it is reused as one of the compared masks, so only
    `runs` additional explainer optimisations are needed. With the default
    (base_mask + runs=1) this compares exactly one pair of masks, the same
    statistic as two fresh runs, at a third less compute per node.
    """
    masks = [base_mask] if base_mask is not None else []
    for _ in range(runs):
        exp = explainer(
            x=x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            index=node_idx,
        )
        masks.append(exp.edge_mask.detach().cpu().float())

    sims = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            similarity = F.cosine_similarity(
                masks[i].unsqueeze(0), masks[j].unsqueeze(0)
            ).item()
            sims.append(similarity)
    return float(np.mean(sims)) if sims else 1.0


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_explanations(model, data, x, n_nodes=20):
    explainer = make_explainer(model)
    node_indices = select_correct_test_nodes(model, data, x, n=n_nodes)
    records = []

    print(f"── Evaluating {len(node_indices)} FastRGCN explanations ─────────────")
    for node_idx in node_indices:
        exp = explainer(
            x=x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            index=node_idx,
        )
        edge_mask = exp.edge_mask

        fp = fidelity_plus(model, data, x, node_idx, edge_mask)
        fm = fidelity_minus(model, data, x, node_idx, edge_mask)
        sp = sparsity(edge_mask)
        stb = stability(explainer, data, x, node_idx, runs=1,
                        base_mask=edge_mask.detach().cpu().float())

        pred = get_pred(model, x, data.edge_index, data.edge_type, node_idx)
        important_edges = int((edge_mask > EDGE_IMPORTANCE_THRESHOLD)
                              .sum().item())
        records.append({
            "node_idx": node_idx,
            "pred": pred,
            "important_edges": important_edges,
            "fidelity+": fp,
            "fidelity-": fm,
            "sparsity": sp,
            "stability": stb,
        })
        print(f"  Node {node_idx:5d} | pred={pred} | "
              f"important={important_edges:4d} | "
              f"F+={fp:.2f} F-={fm:.2f} Spars={sp:.4f} Stab={stb:.4f}")

    df = pd.DataFrame(records)
    print()
    print("Aggregated Metrics")
    for col in ["fidelity+", "fidelity-", "sparsity", "stability"]:
        print(f"  {col:12s}: {df[col].mean():.4f} ± {df[col].std():.4f}")
    print()
    return df


def plot_metrics(df, save_path="plots/eval_metrics.png"):
    metrics = ["fidelity+", "fidelity-", "sparsity", "stability"]
    means = [df[metric].mean() for metric in metrics]
    stds = [df[metric].std() for metric in metrics]
    colors = ["steelblue", "darkorange", "forestgreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(metrics, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", padding=6)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("FastRGCN + GNNExplainer: Explanation Quality Metrics")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved metrics plot to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, data, x = load_model_and_data()
    df = evaluate_explanations(model, data, x, n_nodes=20)
    plot_metrics(df)
    df.to_csv("results/explanation_evaluation.csv", index=False)
    print("✅ Results saved to results/explanation_evaluation.csv")