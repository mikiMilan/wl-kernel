import pickle
from wl_kernel import graph_to_binary_vector


with open("tests/data/FVC2000_DB1_B_101_7.gpickle", "rb") as f:
    G = pickle.load(f)

print(G)

binary_vec = graph_to_binary_vector(G, k=3, size=500)
print(binary_vec)
print("Broj aktivnih pozicija:", binary_vec.sum())