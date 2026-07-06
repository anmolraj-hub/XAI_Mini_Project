"""
Step 3: Model Training & Evaluation

"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config
DATA_ROOT    = "./data/pyg_aifb"
HIDDEN_DIM   = 16
EMB_DIM      = 32        
EPOCHS       = 50
LR           = 0.01
WEIGHT_DECAY = 0.0005
SEED         = 42
MODEL_PATH   = "./models/model_rgcn.pt"

torch.manual_seed(SEED)
torch.set_num_threads(1)                                  # reproducible CPU runs
torch.use_deterministic_algorithms(True, warn_only=True)  # reproducible CPU runs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Load dataset
def load_dataset():
    dataset       = Entities(root=DATA_ROOT, name="AIFB")
    data          = dataset[0].to(device)
    num_relations = int(data.edge_type.max().item()) + 1
    num_classes   = int(data.train_y.max().item()) + 1

    print(f"Dataset loaded: AIFB")
    print(f"   Nodes          : {data.num_nodes:,}")
    print(f"   Edges          : {data.edge_index.shape[1]:,}")
    print(f"   Relation types : {num_relations}")
    print(f"   Classes        : {num_classes}")
    print(f"   Train nodes    : {len(data.train_idx)}")
    print(f"   Test nodes     : {len(data.test_idx)}")
    print()
    return data, num_classes, num_relations


# FastRGCN with learnable embedding
class FastRGCN(torch.nn.Module):
    """
    2-layer FastRGCN with learnable node embeddings.
    FastRGCNConv is compatible with GNNExplainer unlike RGCNConv.
    """
    def __init__(self, num_nodes, emb_dim, hidden_dim,
                 out_channels, num_relations):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, emb_dim)
        self.conv1     = FastRGCNConv(emb_dim, hidden_dim, num_relations)
        self.conv2     = FastRGCNConv(hidden_dim, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type=None):
        # x here is node indices [0, 1, ..., N-1]
        x = self.embedding(x)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


# Training loop
def train(model, data, node_ids, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out  = model(node_ids, data.edge_index, data.edge_type)
    loss = criterion(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, node_ids, idx, labels):
    model.eval()
    out   = model(node_ids, data.edge_index, data.edge_type)
    preds = out[idx].argmax(dim=1).cpu().numpy()
    truth = labels.cpu().numpy()
    acc   = accuracy_score(truth, preds)
    f1    = f1_score(truth, preds, average="macro", zero_division=0)
    return acc, f1, preds, truth


# Full training run
def run_training(data, num_classes, num_relations):
    # Node indices as input
    node_ids = torch.arange(data.num_nodes, device=device)

    model = FastRGCN(
        num_nodes=data.num_nodes,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        out_channels=num_classes,
        num_relations=num_relations,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, train_accs, test_accs = [], [], []

    print("Training FastRGCN")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, data, node_ids, optimizer, criterion)
        tr_acc, _, _, _ = evaluate(model, data, node_ids,
                                   data.train_idx, data.train_y)
        te_acc, _, _, _ = evaluate(model, data, node_ids,
                                   data.test_idx, data.test_y)

        train_losses.append(loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss {loss:.4f} | "
                  f"Train Acc {tr_acc:.4f} | Test Acc {te_acc:.4f}")
    print()
    return model, node_ids, train_losses, train_accs, test_accs


# Final evaluation
def final_evaluation(model, data, node_ids):
    tr_acc, tr_f1, _, _ = evaluate(model, data, node_ids,
                                    data.train_idx, data.train_y)
    te_acc, te_f1, te_preds, te_truth = evaluate(model, data, node_ids,
                                                   data.test_idx, data.test_y)
    print("Final Evaluation")
    print(f"  Train  Accuracy : {tr_acc:.4f}   Macro-F1 : {tr_f1:.4f}")
    print(f"  Test   Accuracy : {te_acc:.4f}   Macro-F1 : {te_f1:.4f}")
    print()
    print("Classification Report (Test set):")
    class_names = None
    try:
        import json
        with open("./data/mappings.json", encoding="utf-8") as handle:
            class_names = json.load(handle)["class_names"]
    except (OSError, KeyError, ValueError):
        print("  (data/mappings.json not found - run build_mappings.py to "
              "get verified research-group names; showing class indices)")
    if class_names is not None:
        present = sorted({int(v) for v in te_truth} |
                         {int(v) for v in te_preds})
        print(classification_report(
            te_truth, te_preds, zero_division=0, labels=present,
            target_names=[class_names[i] for i in present]))
    else:
        print(classification_report(te_truth, te_preds, zero_division=0))


# Plot
def plot_training(train_losses, train_accs, test_accs,
                  save_path="plots/training_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, color="firebrick")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")

    ax2.plot(epochs, train_accs, label="Train Acc", color="steelblue")
    ax2.plot(epochs, test_accs,  label="Test Acc",  color="darkorange")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


# Main
if __name__ == "__main__":
    data, num_classes, num_relations = load_dataset()

    # The AIFB test set has only 36 nodes, so a single run's accuracy swings
    # by ~2.8 points per node. We therefore train over several seeds, report
    # mean +/- std, and save the model of the MEDIAN-accuracy seed (a fair,
    # representative checkpoint) for the explanation pipeline.
    SEEDS = [0, 1, 2, 42, 123]
    runs = []
    for seed in SEEDS:
        print(f"Seed {seed}")
        torch.manual_seed(seed)
        model, node_ids, losses, tr_accs, te_accs = run_training(
            data, num_classes, num_relations)
        te_acc, te_f1, _, _ = evaluate(model, data, node_ids,
                                       data.test_idx, data.test_y)
        print(f"  Seed {seed}: Test Acc {te_acc:.4f} | Macro-F1 {te_f1:.4f}\n")
        runs.append({"seed": seed, "acc": te_acc, "f1": te_f1,
                     "model": model, "node_ids": node_ids,
                     "curves": (losses, tr_accs, te_accs)})

    accs = [run["acc"] for run in runs]
    f1s = [run["f1"] for run in runs]
    mean_acc = sum(accs) / len(accs)
    std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
    mean_f1 = sum(f1s) / len(f1s)
    std_f1 = (sum((f - mean_f1) ** 2 for f in f1s) / len(f1s)) ** 0.5

    print("Summary over seeds")
    for run in runs:
        print(f"  seed {run['seed']:>3}: acc {run['acc']:.4f} "
              f"| macro-F1 {run['f1']:.4f}")
    print(f"  Test Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Test Macro-F1 : {mean_f1:.4f} ± {std_f1:.4f}")
    print()

    # Median-accuracy seed = representative model (no test-set cherry-picking)
    chosen = sorted(runs, key=lambda run: run["acc"])[len(runs) // 2]
    print(f"Saving model of median seed {chosen['seed']} "
          f"(test acc {chosen['acc']:.4f})")
    final_evaluation(chosen["model"], data, chosen["node_ids"])
    plot_training(*chosen["curves"])

    torch.save({
        "model_state":    chosen["model"].state_dict(),
        "num_classes":    num_classes,
        "num_relations":  num_relations,
        "num_nodes":      data.num_nodes,
        "emb_dim":        EMB_DIM,
        "seed":           chosen["seed"],
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")