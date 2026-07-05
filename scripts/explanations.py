"""
XAI Mini Project - Step 4: Human-readable explanations
AIFB Dataset | Strategy 1: GNN Explainability

This script explains the trained FastRGCN model from model_training.py.
It produces:
  1. A global explanation of the most important relation types.
  2. Local explanations for real AIFB person nodes.
  3. A Markdown report with plain-English text you can use in a presentation.
  4. PNG subgraph visualisations for each explained person.
"""

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "./.matplotlib_cache")

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rdflib as rdf
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import FastRGCNConv
from torch_geometric.utils import k_hop_subgraph


# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "./data/pyg_aifb"
MODEL_PATH = "./models/model_rgcn.pt"
MAPPINGS_PATH = "./data/mappings.json"
FULL_GRAPH_PATH = "./data/aifbfixed_complete.n3"
TEST_SET_PATH = "./data/pyg_aifb/aifb/raw/testSet.tsv"
TRAIN_SET_PATH = "./data/pyg_aifb/aifb/raw/trainingSet.tsv"
REPORT_PATH = "results/person_explanations_report.md"
CSV_PATH = "results/person_explanations_summary.csv"

HIDDEN_DIM = 16
SEED = 42
LOCAL_EXPLAINER_EPOCHS = 200
GLOBAL_EXPLAINER_EPOCHS = 100
N_LOCAL_PERSONS = 5
# Persons shown in the report. Pinned so that every re-run reproduces exactly
# the same local-explanation figures. If any name is missing from the test set
# (e.g. after a dataset change), the code falls back to automatic selection.
PINNED_PERSONS = ["Andreas Oberweis", "Maik Herfurth", "Sebastian Blohm",
                  "Peter Bungert", "Saartje Brockmans"]
TOP_LOCAL_EDGES = 10
TOP_GLOBAL_RELATIONS = 10
EDGE_IMPORTANCE_THRESHOLD = 0.5

torch.manual_seed(SEED)
torch.set_num_threads(1)                                  # reproducible CPU runs
torch.use_deterministic_algorithms(True, warn_only=True)  # reproducible CPU runs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Research-group names per class index. PyG's default preprocessing assigns
# class indices in random set order, so hardcoding this list was wrong; it is
# now loaded from data/mappings.json (written by build_mappings.py), which
# stores the exact ordering used when the dataset was built.
CLASS_NAMES = []


# ── Model definitions ─────────────────────────────────────────────────────────
class FastRGCN(torch.nn.Module):
    """Same architecture as model_training.py, used to load saved weights."""

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
    """
    Wrapper for GNNExplainer.
    It receives the pre-computed node embedding matrix as 2D input.
    """

    def __init__(self, conv1, conv2):
        super().__init__()
        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, x, edge_index, edge_type=None):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index, edge_type)


# ── Loading helpers ───────────────────────────────────────────────────────────
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

    print("✅ FastRGCN model loaded.")
    print(f"   Nodes          : {data.num_nodes:,}")
    print(f"   Edges          : {data.edge_index.shape[1]:,}")
    print(f"   Relation types : {num_relations}")
    print()
    return model, full_model, data, x


def _read_target_split(path):
    """Reads AIFB train/test TSV and maps PyG node id -> person URI."""
    mapping = {}
    labels = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            node_id = int(float(row["id"]))
            mapping[node_id] = row["person"]
            labels[node_id] = row["label_affiliation"]
    return mapping, labels


def _read_target_rows(path):
    """Reads AIFB target rows in file order."""
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append({
                "original_id": int(float(row["id"])),
                "person": row["person"],
                "label_affiliation": row["label_affiliation"],
            })
    return rows


def _local_name_from_uri(uri):
    raw = uri.rstrip("/").split("/")[-1].split("#")[-1]
    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)
    raw = raw.replace("_", " ").replace("-", " ")
    return _clean_label(raw or uri)


def _clean_label(text):
    """Fixes common mojibake in AIFB literals, for example JÃ¼rgen -> Jürgen.

    The broken strings are UTF-8 bytes that were decoded as cp1252/latin1,
    so re-encoding with cp1252 (which, unlike latin1, also covers characters
    such as 0x9C) and decoding as UTF-8 restores the original text.
    """
    text = str(text)
    if "Ã" in text or "Â" in text:
        for codec in ("cp1252", "latin1"):
            try:
                return text.encode(codec).decode("utf-8")
            except UnicodeError:
                continue
    return text


def build_metadata():
    """
    Builds readable labels for persons, graph nodes, and relation types.

    The target split files are the most reliable source for person node ids.
    RDF literals are used to replace raw URIs with names and publication titles.
    """
    test_rows = _read_target_rows(TEST_SET_PATH)
    train_rows = _read_target_rows(TRAIN_SET_PATH)
    test_people, test_affiliations = _read_target_split(TEST_SET_PATH)
    train_people, train_affiliations = _read_target_split(TRAIN_SET_PATH)
    target_people = {**train_people, **test_people}
    target_affiliations = {**train_affiliations, **test_affiliations}

    full_graph = rdf.Graph()
    full_graph.parse(FULL_GRAPH_PATH)

    name_pred = rdf.URIRef("http://swrc.ontoware.org/ontology#name")
    title_pred = rdf.URIRef("http://swrc.ontoware.org/ontology#title")

    uri_to_label = {}
    for subject, _, obj in full_graph.triples((None, name_pred, None)):
        uri_to_label[str(subject)] = _clean_label(obj)
    for subject, _, obj in full_graph.triples((None, title_pred, None)):
        uri_to_label.setdefault(str(subject), _clean_label(obj))

    mappings = load_mappings()
    global CLASS_NAMES
    CLASS_NAMES = list(mappings["class_names"])
    uri_to_id = {uri: idx for idx, uri in enumerate(mappings["node_uri"])}

    # NOTE: the 'id' column in the split TSVs is a small sequential id from
    # the original dataset release, NOT a graph node id. Person labels are
    # therefore keyed via the verified URI -> node-id mapping; using the TSV
    # ids would overwrite the labels of unrelated (mostly literal) nodes.
    person_labels = {
        uri_to_id[row["person"]]:
            uri_to_label.get(row["person"],
                             _local_name_from_uri(row["person"]))
        for row in train_rows + test_rows
        if row["person"] in uri_to_id
    }

    node_labels = build_node_labels(uri_to_label, mappings)
    node_labels.update(person_labels)
    relation_labels = build_relation_labels(mappings)
    relation_descriptions = describe_relations(relation_labels)

    print(f"✅ Loaded exact id mappings from {MAPPINGS_PATH}.")
    print(f"✅ Class names: {CLASS_NAMES}")
    print(f"✅ Loaded {len(person_labels)} target person labels.")
    print(f"✅ Loaded {len(relation_labels)} relation labels.")
    print()
    return {
        "test_people": test_people,
        "test_rows": test_rows,
        "train_rows": train_rows,
        "target_affiliations": target_affiliations,
        "person_labels": person_labels,
        "uri_to_label": uri_to_label,
        "node_labels": node_labels,
        "relation_labels": relation_labels,
        "relation_descriptions": relation_descriptions,
    }


def load_mappings():
    """Loads the exact node/relation/class mappings for the processed data.

    PyG's default preprocessing assigns node ids and class ids in random set
    order, which cannot be reconstructed after the fact. build_mappings.py
    rebuilds the dataset deterministically and stores the exact ordering, so
    the labels shown in explanations are guaranteed to be correct.
    """
    path = Path(MAPPINGS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{MAPPINGS_PATH} not found. Run 'python build_mappings.py' and "
            "then re-train with 'python model_training.py' first. Without "
            "this mapping the node and class labels in the explanations "
            "would be arbitrary (PyG assigns ids in random set order)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_node_labels(uri_to_label, mappings):
    """Readable label for every graph node, from the exact id->URI mapping."""
    labels = {}
    for idx, uri in enumerate(mappings["node_uri"]):
        if not str(uri).strip():
            # Shared empty-string literal (e.g., ':fax ""'), a high-degree hub
            labels[idx] = "(empty literal)"
        else:
            labels[idx] = uri_to_label.get(uri, _local_name_from_uri(uri))
    return labels


def build_relation_labels(mappings):
    """Maps PyG edge-type integer -> readable predicate name.

    Edge types 2i and 2i+1 are the forward and inverse direction of the
    i-th relation, matching the Entities preprocessing.
    """
    labels = {}
    for idx, rel in enumerate(mappings["relation_uri"]):
        short = str(rel).split("#")[-1].split("/")[-1]
        labels[2 * idx] = short
        labels[2 * idx + 1] = f"{short}_inverse"
    return labels


def describe_relations(relation_labels):
    descriptions = {}
    for rel_id, label in relation_labels.items():
        base = label.replace("_inverse", "")
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", base).replace("_", " ")
        if label.endswith("_inverse"):
            text = f"inverse of {text}"
        descriptions[rel_id] = text
    return descriptions


# ── Prediction and node selection ─────────────────────────────────────────────
@torch.no_grad()
def predict_classes(full_model, data):
    full_model.eval()
    node_ids = torch.arange(data.num_nodes, device=device)
    logits = full_model(node_ids, data.edge_index, data.edge_type)
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    return predictions, probabilities


def select_person_nodes(full_model, data, metadata, n=N_LOCAL_PERSONS):
    """Selects correctly classified test-set persons where possible."""
    predictions, probabilities = predict_classes(full_model, data)
    person_labels = metadata["person_labels"]
    test_rows = metadata["test_rows"]
    uri_to_label = metadata["uri_to_label"]
    node_labels = metadata["node_labels"]

    selected = []
    for offset, node_tensor in enumerate(data.test_idx.detach().cpu()):
        node_id = int(node_tensor.item())
        person_uri = test_rows[offset]["person"] if offset < len(test_rows) else ""
        person_name = uri_to_label.get(person_uri, _local_name_from_uri(person_uri))
        if not person_name:
            person_name = person_labels.get(node_id, f"Person node {node_id}")

        expected = int(data.test_y[offset].item())
        predicted = int(predictions[node_id].item())
        confidence = float(probabilities[node_id, predicted].item())
        person_labels[node_id] = person_name
        node_labels[node_id] = person_name
        selected.append({
            "node_id": node_id,
            "name": person_name,
            "uri": person_uri,
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "correct": predicted == expected,
        })

    # Prefer the pinned report persons so figures are identical on every run.
    pinned = [row for row in selected if row["name"] in PINNED_PERSONS]
    if len(pinned) == len(PINNED_PERSONS):
        selected = sorted(pinned,
                          key=lambda row: PINNED_PERSONS.index(row["name"]))
    else:
        selected.sort(key=lambda row: (not row["correct"], -row["confidence"]))
        selected = selected[:n]

    print("✅ Selected people to explain:")
    for row in selected:
        pred_name = class_name(row["predicted"])
        marker = "correct" if row["correct"] else "model differs from label"
        print(f"   node {row['node_id']:>4} | {row['name']} | "
              f"{pred_name} | {marker}")
    print()
    return selected


def class_name(class_idx):
    if 0 <= class_idx < len(CLASS_NAMES):
        return CLASS_NAMES[class_idx]
    return f"class_{class_idx}"


def receptive_field_edge_mask(node_id, data, num_hops=2):
    """Boolean mask of edges inside the k-hop receptive field of node_id.

    The trained model has two FastRGCN layers, so only edges within the
    2-hop neighbourhood of the target node can influence its prediction.
    GNNExplainer optimises a mask over *all* edges, and mask values outside
    the receptive field are pure optimisation noise; they must be removed
    before ranking edges for the explanation.
    """
    _, _, _, edge_mask = k_hop_subgraph(
        int(node_id), num_hops, data.edge_index,
        num_nodes=data.num_nodes, flow="source_to_target")
    return edge_mask


# ── Explainers ────────────────────────────────────────────────────────────────
def make_explainer(model, epochs):
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


def global_explanation(model, data, x, metadata,
                       save_path="plots/global_explanation.png"):
    """
    Averages GNNExplainer edge masks across training nodes and reports which
    relation types matter most overall.
    """
    print("── Global Explanation ────────────────────────────────")
    explainer = make_explainer(model, GLOBAL_EXPLAINER_EPOCHS)

    edge_type = data.edge_type.detach().cpu().numpy()
    num_rels = int(edge_type.max()) + 1
    relation_scores = np.zeros(num_rels)
    relation_counts = np.zeros(num_rels)

    train_nodes = data.train_idx[:20].tolist()
    print(f"  Running GNNExplainer on {len(train_nodes)} training nodes...")

    for index, node_idx in enumerate(train_nodes, start=1):
        exp = explainer(
            x=x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            index=int(node_idx),
        )
        edge_mask = exp.edge_mask.detach().cpu().numpy()
        for rel_id, importance in zip(edge_type, edge_mask):
            relation_scores[rel_id] += importance
            relation_counts[rel_id] += 1

        if index % 5 == 0:
            print(f"  Processed {index}/{len(train_nodes)} nodes...")

    relation_average = np.where(
        relation_counts > 0,
        relation_scores / relation_counts,
        0,
    )
    top_indices = np.argsort(relation_average)[-TOP_GLOBAL_RELATIONS:][::-1]

    relation_labels = metadata["relation_labels"]
    print(f"\n  Top {TOP_GLOBAL_RELATIONS} relation types:")
    for rank, rel_id in enumerate(top_indices, start=1):
        label = relation_labels.get(int(rel_id), f"relation_{rel_id}")
        print(f"  {rank:>2}. {label:<32} {relation_average[rel_id]:.4f}")

    names = [relation_labels.get(int(rel_id), f"relation_{rel_id}")
             for rel_id in top_indices]
    scores = [relation_average[rel_id] for rel_id in top_indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set_xlabel("Average edge-mask importance")
    ax.set_title("Global Explanation: Most Important Relation Types")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Global explanation saved to {save_path}\n")
    return top_indices, relation_average


def run_local_explanations(model, data, x, people):
    print("── Local Person Explanations ─────────────────────────")
    explainer = make_explainer(model, LOCAL_EXPLAINER_EPOCHS)
    explanations = []

    for person in people:
        node_id = person["node_id"]
        exp = explainer(
            x=x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            index=node_id,
        )
        in_field = receptive_field_edge_mask(node_id, data)
        masked = exp.edge_mask.detach() * in_field.to(
            exp.edge_mask.device).float()
        important_edges = int((masked > EDGE_IMPORTANCE_THRESHOLD)
                              .sum().item())
        person = dict(person)
        person["important_edges"] = important_edges
        explanations.append((person, exp))

        print(f"  {person['name']} (node {node_id})")
        print(f"     predicted: {class_name(person['predicted'])}")
        print(f"     confidence: {person['confidence']:.3f}")
        print(f"     important edges above {EDGE_IMPORTANCE_THRESHOLD}: "
              f"{important_edges}")
    print()
    return explanations


# ── Explanation summaries ─────────────────────────────────────────────────────
def extract_top_edges(exp, data, metadata, node_id, top_k=TOP_LOCAL_EDGES):
    edge_mask = exp.edge_mask.detach().cpu()
    in_field = receptive_field_edge_mask(node_id, data).cpu()
    edge_mask = (edge_mask * in_field.float()).numpy()
    edge_index = data.edge_index.detach().cpu().numpy()
    edge_type = data.edge_type.detach().cpu().numpy()

    node_labels = metadata["node_labels"]
    relation_labels = metadata["relation_labels"]
    relation_descriptions = metadata["relation_descriptions"]

    ranked = np.argsort(edge_mask)[-top_k:][::-1]
    rows = []
    for edge_pos in ranked:
        if edge_mask[edge_pos] <= 0.0:
            continue  # outside the receptive field
        source = int(edge_index[0][edge_pos])
        target = int(edge_index[1][edge_pos])
        relation_id = int(edge_type[edge_pos])
        rows.append({
            "source_id": source,
            "source": node_labels.get(source, f"node_{source}"),
            "target_id": target,
            "target": node_labels.get(target, f"node_{target}"),
            "relation_id": relation_id,
            "relation": relation_labels.get(relation_id,
                                            f"relation_{relation_id}"),
            "relation_text": relation_descriptions.get(
                relation_id, f"relation {relation_id}"),
            "importance": float(edge_mask[edge_pos]),
        })
    return rows


def make_plain_english_explanation(person, top_edges):
    relation_counts = Counter(edge["relation_text"] for edge in top_edges[:5])
    relation_phrase = ", ".join(
        f"{name} ({count})" if count > 1 else name
        for name, count in relation_counts.most_common(3)
    )
    strongest = top_edges[0] if top_edges else None
    prediction = class_name(person["predicted"])

    if strongest is None:
        return (
            f"For {person['name']}, the model predicts {prediction}. "
            "GNNExplainer did not isolate a strong individual edge, so this "
            "case should be presented as a low-specificity explanation."
        )

    return (
        f"For {person['name']}, the model predicts the research group "
        f"{prediction}. The explanation says this decision is mainly supported "
        f"by graph connections of type {relation_phrase}. The strongest single "
        f"piece of evidence is the connection '{strongest['source']}' "
        f"-- {strongest['relation_text']} --> '{strongest['target']}' "
        f"with importance {strongest['importance']:.2f}. In simple terms, the "
        "model is looking at this person's surrounding publications, projects, "
        "topics, and related entities, then using the most influential links "
        "around that person to justify the group prediction."
    )


def save_report(explanation_rows, global_top, global_scores, metadata,
                path=REPORT_PATH):
    relation_labels = metadata["relation_labels"]

    lines = [
        "# Person-Level GNNExplanations for AIFB",
        "",
        "This report explains the trained FastRGCN model in presentation-ready "
        "language. Each person section states the model prediction and the "
        "graph relations that GNNExplainer considered most important.",
        "",
        "## Global Explanation",
        "",
        "Across the sampled training nodes, these relation types had the highest "
        "average importance:",
        "",
    ]

    for rank, rel_id in enumerate(global_top, start=1):
        label = relation_labels.get(int(rel_id), f"relation_{rel_id}")
        lines.append(f"{rank}. **{label}** - importance "
                     f"{global_scores[rel_id]:.4f}")

    lines.extend(["", "## Local Person Explanations", ""])

    for row in explanation_rows:
        person = row["person"]
        lines.append(f"### {person['name']} (node {person['node_id']})")
        lines.append("")
        lines.append(f"- Predicted group: **{class_name(person['predicted'])}**")
        lines.append(f"- Model confidence: **{person['confidence']:.3f}**")
        lines.append(f"- Ground-truth match: **{person['correct']}**")
        lines.append(f"- Edges above importance threshold "
                     f"{EDGE_IMPORTANCE_THRESHOLD} (within 2-hop receptive "
                     f"field): **{person['important_edges']}**")
        lines.append(f"- Table and figure show the top "
                     f"{len(row['top_edges'])} edges by importance")
        lines.append("")
        lines.append(row["plain_text"])
        lines.append("")
        lines.append("| Rank | Source | Relation | Target | Importance |")
        lines.append("|---:|---|---|---|---:|")
        for rank, edge in enumerate(row["top_edges"], start=1):
            lines.append(
                f"| {rank} | {edge['source']} | {edge['relation']} | "
                f"{edge['target']} | {edge['importance']:.3f} |"
            )
        lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Human-readable report saved to {path}")


def save_csv(explanation_rows, path=CSV_PATH):
    fieldnames = [
        "node_id",
        "person",
        "predicted_group",
        "confidence",
        "correct",
        "important_edges",
        "plain_english_explanation",
        "top_relations",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in explanation_rows:
            person = row["person"]
            top_relations = "; ".join(
                f"{edge['relation']}={edge['importance']:.3f}"
                for edge in row["top_edges"][:5]
            )
            writer.writerow({
                "node_id": person["node_id"],
                "person": person["name"],
                "predicted_group": class_name(person["predicted"]),
                "confidence": f"{person['confidence']:.3f}",
                "correct": person["correct"],
                "important_edges": person["important_edges"],
                "plain_english_explanation": row["plain_text"],
                "top_relations": top_relations,
            })
    print(f"✅ Summary CSV saved to {path}")


# ── Visualisation ─────────────────────────────────────────────────────────────
def _trim(text, max_length=28):
    text = str(text)
    return text if len(text) <= max_length else text[:max_length - 3] + "..."


def safe_filename(text):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_")[:40] or "person"


def visualise_local(person, top_edges, save_path):
    # The graph is keyed on node ids, NOT on (trimmed) labels: two distinct
    # nodes whose labels collide after trimming must not be merged, and
    # parallel edges with different relations must not overwrite each other.
    graph = nx.DiGraph()
    target_id = person["node_id"]
    node_names = {target_id: _trim(person["name"])}

    for edge in top_edges:
        src, dst = edge["source_id"], edge["target_id"]
        node_names.setdefault(src, _trim(edge["source"]))
        node_names.setdefault(dst, _trim(edge["target"]))
        if graph.has_edge(src, dst):
            attrs = graph[src][dst]
            attrs["weight"] = max(attrs["weight"], edge["importance"])
            if edge["relation"] not in attrs["relation"]:
                attrs["relation"] += f" / {edge['relation']}"
        else:
            graph.add_edge(src, dst, weight=edge["importance"],
                           relation=edge["relation"])

    graph.add_node(target_id)

    fig, ax = plt.subplots(figsize=(13, 8))
    pos = nx.spring_layout(graph, seed=SEED, k=2.2)

    node_colors = [
        "gold" if node == target_id else "lightsteelblue"
        for node in graph.nodes()
    ]
    edge_widths = [
        max(0.5, graph[u][v]["weight"] * 6)
        for u, v in graph.edges()
    ]
    edge_colors = [graph[u][v]["weight"] for u, v in graph.edges()]

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                           node_size=2100, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=node_names, font_size=8,
                            font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Oranges,
        edge_vmin=0.0,
        edge_vmax=1.0,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    edge_labels = {
        (u, v): f"{data['relation']} ({data['weight']:.2f})"
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                 font_size=6, ax=ax)

    legend = [
        mpatches.Patch(facecolor="gold", label="Explained person"),
        mpatches.Patch(facecolor="lightsteelblue", label="Neighbour entity"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=10)
    ax.set_title(
        f"Local explanation for {person['name']}\n"
        f"Predicted group: {class_name(person['predicted'])}\n"
        f"Top {len(top_edges)} GNNExplainer edges within the 2-hop "
        f"receptive field",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Local graph saved to {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, full_model, data, x = load_model_and_data()
    metadata = build_metadata()

    global_top, global_scores = global_explanation(model, data, x, metadata)

    people = select_person_nodes(full_model, data, metadata, n=N_LOCAL_PERSONS)
    local_explanations = run_local_explanations(model, data, x, people)

    explanation_rows = []
    saved_explanations = []
    for person, exp in local_explanations:
        top_edges = extract_top_edges(exp, data, metadata,
                                      person["node_id"], TOP_LOCAL_EDGES)
        plain_text = make_plain_english_explanation(person, top_edges)
        explanation_rows.append({
            "person": person,
            "top_edges": top_edges,
            "plain_text": plain_text,
        })
        saved_explanations.append((person["node_id"], exp))

        file_name = f"plots/local_exp_{safe_filename(person['name'])}.png"
        visualise_local(person, top_edges, file_name)

    save_report(explanation_rows, global_top, global_scores, metadata)
    save_csv(explanation_rows)

    torch.save(
        {
            "explanations": saved_explanations,
            "person_summaries": explanation_rows,
            "x": x.detach().cpu(),
        },
        "./models/explanations.pt",
    )
    print("✅ All done. Use person_explanations_report.md for your explanation.")