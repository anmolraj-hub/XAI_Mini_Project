"""Deterministic AIFB preprocessing shared by training, explanation, evaluation."""

import csv
import gzip
import re
from collections import Counter
from pathlib import Path

import rdflib as rdf
import torch
from torch_geometric.data import Data


DATA_ROOT = Path("./data")
RAW_GRAPH_PATH = Path("./data/pyg_aifb/aifb/raw/aifb_stripped.nt.gz")
FULL_GRAPH_PATH = DATA_ROOT / "aifbfixed_complete.n3"
TRAIN_SET_PATH = Path("./data/pyg_aifb/aifb/raw/trainingSet.tsv")
TEST_SET_PATH = Path("./data/pyg_aifb/aifb/raw/testSet.tsv")
CACHE_PATH = Path("./data/aifb_deterministic.pt")

CLASS_URI_ORDER = [
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id1instance",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id2instance",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id3instance",
    "http://www.aifb.uni-karlsruhe.de/Forschungsgruppen/viewForschungsgruppeOWL/id4instance",
]

FALLBACK_CLASS_NAMES = [
    "Business Information & Communication Systems",
    "Efficient Algorithms",
    "Knowledge Management",
    "Complexity Management",
]


def clean_label(text):
    text = str(text)
    if "Ã" in text or "Â" in text:
        try:
            return text.encode("latin1").decode("utf-8")
        except UnicodeError:
            return text
    return text


def short_uri_label(uri):
    raw = str(uri).rstrip("/").split("/")[-1].split("#")[-1]
    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)
    raw = raw.replace("_", " ").replace("-", " ")
    return clean_label(raw or uri)


def read_split_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append({
                "person": row["person"],
                "label_affiliation": row["label_affiliation"],
            })
    return rows


def load_literal_labels():
    graph = rdf.Graph()
    graph.parse(FULL_GRAPH_PATH)

    predicates = [
        rdf.URIRef("http://swrc.ontoware.org/ontology#name"),
        rdf.URIRef("http://swrc.ontoware.org/ontology#title"),
        rdf.URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
    ]

    labels = {}
    for predicate in predicates:
        for subject, _, obj in graph.triples((None, predicate, None)):
            labels.setdefault(str(subject), clean_label(obj))
    return labels


def build_deterministic_aifb():
    graph = rdf.Graph()
    with gzip.open(RAW_GRAPH_PATH, "rb") as handle:
        graph.parse(file=handle, format="nt")

    freq = Counter(graph.predicates())
    relations = sorted(set(graph.predicates()),
                       key=lambda rel: (-freq[rel], str(rel)))
    nodes = sorted({str(node) for node in graph.subjects()} |
                   {str(node) for node in graph.objects()})

    relation_to_id = {relation: idx for idx, relation in enumerate(relations)}
    node_to_id = {uri: idx for idx, uri in enumerate(nodes)}

    edges = []
    for source, predicate, target in graph.triples((None, None, None)):
        src = node_to_id[str(source)]
        dst = node_to_id[str(target)]
        rel = relation_to_id[predicate]
        edges.append([src, dst, 2 * rel])
        edges.append([dst, src, 2 * rel + 1])

    edge = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = len(nodes)
    num_relations = 2 * len(relations)
    sort_key = num_nodes * num_relations * edge[0] + num_relations * edge[1] + edge[2]
    perm = torch.argsort(sort_key)
    edge = edge[:, perm]

    class_uri_to_id = {uri: idx for idx, uri in enumerate(CLASS_URI_ORDER)}

    train_rows = read_split_rows(TRAIN_SET_PATH)
    test_rows = read_split_rows(TEST_SET_PATH)

    train_idx = torch.tensor([node_to_id[row["person"]] for row in train_rows],
                             dtype=torch.long)
    train_y = torch.tensor(
        [class_uri_to_id[row["label_affiliation"]] for row in train_rows],
        dtype=torch.long,
    )
    test_idx = torch.tensor([node_to_id[row["person"]] for row in test_rows],
                            dtype=torch.long)
    test_y = torch.tensor(
        [class_uri_to_id[row["label_affiliation"]] for row in test_rows],
        dtype=torch.long,
    )

    data = Data(
        edge_index=edge[:2],
        edge_type=edge[2],
        train_idx=train_idx,
        train_y=train_y,
        test_idx=test_idx,
        test_y=test_y,
        num_nodes=num_nodes,
    )

    literal_labels = load_literal_labels()
    node_labels = {
        idx: literal_labels.get(uri, short_uri_label(uri))
        for idx, uri in enumerate(nodes)
    }

    relation_labels = {}
    for idx, relation in enumerate(relations):
        short = str(relation).split("#")[-1].split("/")[-1]
        relation_labels[2 * idx] = short
        relation_labels[2 * idx + 1] = f"{short}_inverse"

    class_names = [
        literal_labels.get(uri, fallback)
        for uri, fallback in zip(CLASS_URI_ORDER, FALLBACK_CLASS_NAMES)
    ]

    metadata = {
        "node_to_uri": {idx: uri for idx, uri in enumerate(nodes)},
        "uri_to_node": node_to_id,
        "node_labels": node_labels,
        "relation_labels": relation_labels,
        "class_names": class_names,
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
    return data, metadata


def load_aifb_dataset(device=None, force_rebuild=False):
    if CACHE_PATH.exists() and not force_rebuild:
        payload = torch.load(CACHE_PATH, weights_only=False)
        data = payload["data"]
        metadata = payload["metadata"]
    else:
        data, metadata = build_deterministic_aifb()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"data": data, "metadata": metadata}, CACHE_PATH)

    if device is not None:
        data = data.to(device)
    return data, metadata
