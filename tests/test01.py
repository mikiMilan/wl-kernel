import pickle
from wl_kernel import wl_relabel_custom



with open("tests/data/FVC2000_DB1_B_101_7.gpickle", "rb") as f:
    fingerprint_graph = pickle.load(f)

    print(fingerprint_graph)
    r = wl_relabel_custom(fingerprint_graph, node_labels=["type"])
    print(r)