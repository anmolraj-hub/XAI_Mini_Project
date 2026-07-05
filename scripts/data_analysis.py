"""
Step 2: Data Analysis

"""

import os
import rdflib as rdf
from rdflib import Graph, RDF, RDFS, OWL
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 120

# Paths 
DATA_DIR = "./data"
RDF_FILE = os.path.join(DATA_DIR, "aifbfixed_complete.n3")
# Label file shipped with the DGL zip
LABEL_FILE = os.path.join(DATA_DIR, "completeDataset.tsv")


# Load RDF graph
def load_graph(path: str) -> Graph:
    print(f" Loading RDF graph from {path} ...")
    g = Graph()
    g.parse(path)
    print(f" Loaded {len(g):,} triples.\n")
    return g


# Basic statistics 
def basic_stats(g: Graph) -> dict:
    subjects   = set(g.subjects())
    predicates = set(g.predicates())
    objects    = set(g.objects())
    all_nodes  = subjects | objects

    stats = {
        "Total triples": len(g),
        "Unique subjects": len(subjects),
        "Unique predicates": len(predicates),
        "Unique objects": len(objects),
        "Unique nodes (subj ∪ obj)": len(all_nodes),
    }
    print(" Basic Statistics")
    for k, v in stats.items():
        print(f"  {k:<35} {v:>10,}")
    print()
    return stats


# Top predicates (edge types)
def top_predicates(g: Graph, n: int = 20) -> pd.DataFrame:
    pred_counts = Counter(str(p) for _, p, _ in g)
    top = pred_counts.most_common(n)
    df = pd.DataFrame(top, columns=["Predicate", "Count"])
    # Shorten URIs for readability
    df["Short Name"] = df["Predicate"].apply(
        lambda u: u.split("#")[-1] if "#" in u else u.split("/")[-1]
    )
    print("Top Predicates (Edge Types)")
    print(df[["Short Name", "Count"]].to_string(index=False))
    print()
    return df


# Node types via rdf:type
def node_types(g: Graph) -> pd.DataFrame:
    type_counts = Counter(str(o) for _, p, o in g if p == RDF.type)
    df = pd.DataFrame(type_counts.most_common(30), columns=["Type URI", "Count"])
    df["Short Name"] = df["Type URI"].apply(
        lambda u: u.split("#")[-1] if "#" in u else u.split("/")[-1]
    )
    print("Node Types (via rdf:type)")
    print(df[["Short Name", "Count"]].to_string(index=False))
    print()
    return df


# Class distribution (affiliation labels)
AFFILIATION_URI = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")

def class_distribution(g: Graph) -> pd.DataFrame:
    """Count persons per research group (affiliation predicate)."""
    dst = [str(o) for _, p, o in g.triples((None, AFFILIATION_URI, None))]
    counts = Counter(dst)

    # Map full URI -> human-readable label
    name_pred = rdf.term.URIRef("http://swrc.ontoware.org/ontology#name")
    label_map = {}
    for group_uri in counts:
        grp = rdf.term.URIRef(group_uri)
        for _, _, name in g.triples((grp, name_pred, None)):
            label_map[group_uri] = str(name)
            break

    rows = [
        {
            "Research Group URI": uri,
            "Label": label_map.get(uri, uri.split("/")[-1]),
            "Members": cnt,
        }
        for uri, cnt in counts.most_common()
    ]
    df = pd.DataFrame(rows)
    print("Class Distribution (affiliation)")
    print(df[["Label", "Members"]].to_string(index=False))
    print()
    return df


# Visualisations
def plot_class_distribution(df: pd.DataFrame, save_path: str = "plots/class_distribution.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(df["Label"], df["Members"], color="steelblue")
    ax.bar_label(bars, padding=4)
    ax.set_xlabel("Number of Members")
    ax.set_title("AIFB Dataset – Research Group Distribution")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved figure to {save_path}")


def plot_top_predicates(df: pd.DataFrame, n: int = 15,
                        save_path: str = "plots/top_predicates.png"):
    top = df.head(n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["Short Name"], top["Count"], color="darkorange")
    ax.set_xlabel("Triple Count")
    ax.set_title(f"AIFB Dataset – Top {n} Predicates")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved figure to {save_path}")


# Sample some triples for intuition
def show_sample_triples(g: Graph, predicate_uri: str, n: int = 5):
    pred = rdf.term.URIRef(predicate_uri)
    print(f"Sample triples for <{predicate_uri.split('#')[-1]}>")
    for i, (s, p, o) in enumerate(g.triples((None, pred, None))):
        print(f"  {str(s).split('/')[-1]:40s}  →  {str(o).split('/')[-1]}")
        if i >= n - 1:
            break
    print()


# Main
if __name__ == "__main__":
    g = load_graph(RDF_FILE)

    stats    = basic_stats(g)
    pred_df  = top_predicates(g)
    type_df  = node_types(g)
    class_df = class_distribution(g)

    plot_class_distribution(class_df)
    plot_top_predicates(pred_df)

    show_sample_triples(g, "http://swrc.ontoware.org/ontology#affiliation")
    show_sample_triples(g, "http://swrc.ontoware.org/ontology#publication")