import pickle
import numpy as np
import networkx as nx
from wl_kernel.kernel import WLkernel
from scipy.stats import entropy


with open("tests/data/graph_dataset/G_1000_0.1_orig.gpickle", "rb") as f:
    G1 = pickle.load(f)
with open("tests/data/graph_dataset/G_1000_0.1_mod_100.gpickle", "rb") as f:
    G2 = pickle.load(f)

# Kreiraj WLkernel instance
kernel1 = WLkernel(G1, size=100024)
kernel2 = WLkernel(G2, size=100024)

# Dobij vektore
vec1 = kernel1.degree_vector(iter=1)
# print(vec1)
vec2 = kernel2.degree_vector(iter=1)
# print(vec1)

probs = vec1 / np.sum(vec1)
ent = entropy(probs)
print("Entropy: ", ent)

# Izračunaj i ispiši sve metrike
for method in ["cosine", "jaccard"]:
    sim = WLkernel.similarity(vec1, vec2, method=method)
    print(f"{method.title()} similarity: {sim:.4f}")