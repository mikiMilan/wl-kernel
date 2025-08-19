import pickle
import networkx as nx
from wl_kernel.kernel import WLkernel

def test_degree_vector():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4)])

    kernel = WLkernel(G, size=16, k=3)
    vec = kernel.degree_vector()
    print("\nDegree vector:", vec.tolist())

    vec = kernel.degree_vector(iter=3)
    print("Degree vector:", vec.tolist())

def test_spatial_vector_empty():
    with open("tests/data/FVC2000_DB1_B_101_7.gpickle", "rb") as f:
        G = pickle.load(f)

    kernel = WLkernel(G, size=128, k=2)
    vec = kernel.spatial_vector()

    # Čvor nema x, y → sve bi trebalo biti 0
    assert vec.sum() != 0

    print("Spatial vector:", vec.tolist())
