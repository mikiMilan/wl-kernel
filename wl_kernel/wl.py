import networkx as nx
import hashlib
import math

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def hash_str_to_int(s, base):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % base

def wl_relabel_custom(G: nx.Graph, h: int = 2, base: int = 200, node_labels: list = []) -> dict:
    """
    WL relabeling with spatial awareness and modular hashing.
    
    Args:
        G: Graph with node positions: G.nodes[n]['pos'] = (x, y)
        h: Number of iterations
        base: Size of output label space (mod hash)
    
    Returns:
        Dict: node → label (int in [0, base))
    """
    # Step 1: Inicijalne oznake (po modulu 2)
    labels = {}
    for node in G.nodes():
        label1 = "" if len(node_labels) > 0 else G.degree[node]
        for node_lab in node_labels:
            label1 += str(G.nodes[node].get(node_lab))

        if type(label1) == str:
            label1 = hash_str_to_int(label1, 2) + 1
        else:
            label1 = (label1 % 2) + 1

        
        labels[node] = [label1, None]

    
    '''
    # Step 2: Iteracije
    for i in range(h):
        new_labels = {}
        for node in G.nodes():
            center = G.nodes[node].get("pos", (0, 0))
            neighbors = list(G.neighbors(node))
            # sortiranje po udaljenosti
            neighbors.sort(key=lambda n: euclidean(center, G.nodes[n].get("pos", (0, 0))))
            # konstruisanje stringa
            neighbor_labels = [str(labels[n]) for n in neighbors]
            s = f"{labels[node]}|" + "|".join(neighbor_labels)
            # hash mod base
            new_labels[node] = hash_str_to_int(s, base)
        labels = new_labels  # ažuriranje
    '''
    return labels
