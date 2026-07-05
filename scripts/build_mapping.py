"""
Step 1b: Deterministic dataset build + label mappings

"""

import gzip
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import rdflib as rdf
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Entities

DATA_ROOT = "./data/pyg_aifb"
MAPPINGS_PATH = "./data/mappings.json"

# Official AIFB research-group names, keyed by the local name of the affiliation URI 
GROUP_NAMES = {
    "id1instance": "Business Information & Communication Systems",
    "id2instance": "Efficient Algorithms",
    "id3instance": "Knowledge Management",
    "id4instance": "Complexity Management",
    "id5instance": "Usability Engineering",
}


class DeterministicAIFB(Entities):
    """``Entities`` subclass whose ``process()`` uses sorted, reproducible
    orderings for nodes, relations, and class labels.

    The edge construction mirrors PyG's original implementation exactly
    (including the ``str()`` node keys, the inverse-relation ``2r / 2r+1``
    scheme, and the final edge sort), so every downstream script can keep
    loading the dataset through the plain ``Entities`` class.
    """

    captured_mappings = None

    def process(self):
        graph_file, task_file, train_file, test_file = self.raw_paths

        graph = rdf.Graph()
        with gzip.open(graph_file, "rb") as handle:
            graph.parse(file=handle, format="nt")

        # Deterministic relation order: frequency (desc), then URI string.
        # PyG sorts by frequency only, leaving ties in random set order.
        freq = Counter(graph.predicates())
        relations = sorted(set(graph.predicates()),
                           key=lambda p: (-freq[p], str(p)))
        nodes = sorted({str(s) for s in graph.subjects()} |
                       {str(o) for o in graph.objects()})

        num_nodes = len(nodes)
        num_edge_types = 2 * len(relations)
        relations_dict = {rel: i for i, rel in enumerate(relations)}
        nodes_dict = {node: i for i, node in enumerate(nodes)}

        edges = []
        for s, p, o in graph.triples((None, None, None)):
            src, dst = nodes_dict[str(s)], nodes_dict[str(o)]
            rel = relations_dict[p]
            edges.append([src, dst, 2 * rel])
            edges.append([dst, src, 2 * rel + 1])

        edge = torch.tensor(edges, dtype=torch.long).t().contiguous()
        sort_key = (num_nodes * num_edge_types * edge[0]
                    + num_edge_types * edge[1] + edge[2])
        edge = edge[:, sort_key.argsort()]
        edge_index, edge_type = edge[:2], edge[2]

        # Deterministic class order: sorted affiliation URIs
        # (id1instance < id2instance < ... < id5instance).
        labels_df = pd.read_csv(task_file, sep="\t")
        class_uris = sorted(set(labels_df["label_affiliation"]))
        labels_dict = {lab: i for i, lab in enumerate(class_uris)}

        def read_split(path):
            split_df = pd.read_csv(path, sep="\t")
            idx = [nodes_dict[person] for person in split_df["person"]]
            y = [labels_dict[lab] for lab in split_df["label_affiliation"]]
            return (torch.tensor(idx, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long))

        train_idx, train_y = read_split(train_file)
        test_idx, test_y = read_split(test_file)

        data = Data(edge_index=edge_index, edge_type=edge_type,
                    train_idx=train_idx, train_y=train_y,
                    test_idx=test_idx, test_y=test_y, num_nodes=num_nodes)

        if self.hetero:
            data = data.to_heterogeneous(node_type_names=["v"])

        self.save([data], self.processed_paths[0])

        DeterministicAIFB.captured_mappings = {
            "build": "deterministic-v1",
            "node_uri": nodes,
            "relation_uri": [str(rel) for rel in relations],
            "class_uri": class_uris,
            "class_names": [
                GROUP_NAMES.get(uri.rstrip("/").split("/")[-1], uri)
                for uri in class_uris
            ],
        }


def verify(data, mappings):
    """Sanity checks proving the saved mapping matches the saved data."""
    assert len(mappings["node_uri"]) == data.num_nodes, \
        "node mapping size does not match the graph"

    num_edge_types = int(data.edge_type.max()) + 1
    assert len(mappings["relation_uri"]) * 2 == num_edge_types, \
        "relation mapping size does not match the edge types"

    # The class distribution of test_y must match the official testSet.tsv.
    test_df = pd.read_csv(Path(DATA_ROOT) / "aifb" / "raw" / "testSet.tsv",
                          sep="\t")
    expected = Counter(test_df["label_affiliation"])
    class_uris = mappings["class_uri"]
    actual = Counter(class_uris[int(y)] for y in data.test_y)
    assert actual == expected, (
        f"class mapping mismatch:\n  expected {dict(expected)}\n"
        f"  actual   {dict(actual)}")

    print("Verification passed:")
    print(f"   {data.num_nodes:,} nodes mapped")
    print(f"   {len(mappings['relation_uri'])} relations "
          f"({num_edge_types} edge types) mapped")
    for uri in class_uris:
        name = GROUP_NAMES.get(uri.rstrip("/").split("/")[-1], uri)
        print(f"   class {class_uris.index(uri)} = {name} "
              f"({expected.get(uri, 0)} test nodes)")
    print()


def main():
    processed_dir = Path(DATA_ROOT) / "aifb" / "processed"
    if processed_dir.exists():
        print(f"Deleting old (non-deterministic) processed data: "
              f"{processed_dir}")
        shutil.rmtree(processed_dir)

    print("Rebuilding AIFB with deterministic node/relation/class ids")
    dataset = DeterministicAIFB(root=DATA_ROOT, name="AIFB")
    data = dataset[0]

    mappings = DeterministicAIFB.captured_mappings
    assert mappings is not None, "process() was not triggered"

    verify(data, mappings)

    Path(MAPPINGS_PATH).write_text(
        json.dumps(mappings, ensure_ascii=False), encoding="utf-8")
    print(f"Mappings saved to {MAPPINGS_PATH}")
    print()
   


if __name__ == "__main__":
    main()