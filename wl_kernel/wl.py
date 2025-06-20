import math
import networkx as nx
from collections import Counter
import hashlib
import numpy as np


def stable_hash_bin(label, size=500):
    """
    Hash string label u indeks 0..(size-1) pomoću md5
    """
    h = hashlib.md5(label.encode('utf-8')).hexdigest()
    return int(h, 16) % size

def graph_to_binary_vector(G, k=3, size=500):
    vec = np.zeros(size, dtype=np.uint8)

    for v in G.nodes:
        label = get_spatial_label(v, G, k=k)
        if label == "empty" or label == "NA":
            continue
        idx = stable_hash_bin(label, size)
        vec[idx] = 1

    return vec

def fingerprint_vector(G, k=3):
    labels = [get_spatial_label(v, G, k=k) for v in G.nodes]
    return Counter(labels)

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle_between(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx)) % 360

def quantize(value, base):
    return round(value / base) * base

def get_spatial_label(v, G, k=3, dist_bin=10, angle_bin=10):
    if 'x' not in G.nodes[v] or 'y' not in G.nodes[v]:
        return "NA"

    x0, y0 = G.nodes[v]['x'], G.nodes[v]['y']
    pos_v = (x0, y0)

    neighbors = []
    for u in G.neighbors(v):
        if 'x' not in G.nodes[u] or 'y' not in G.nodes[u]:
            continue
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        pos_u = (x1, y1)
        dist = euclidean(pos_v, pos_u)
        angle = angle_between(pos_v, pos_u)
        neighbors.append((u, dist, angle))

    # Sort by score descending
    neighbors.sort(key=lambda tup: tup[1])
    top_k = neighbors[:k]

    if not top_k:
        return "empty"

    # Bazni ugao = ugao ka prvom susjedu
    base_angle = top_k[0][2]

    parts = []
    for i, (u, dist, angle) in enumerate(top_k):
        rel_angle = (angle - base_angle) % 360
        dist_q = quantize(dist, dist_bin)
        
        if i == 0:
            parts.append(f"{int(dist_q / 10)}")  # bez ugla
        else:
            angle_q = quantize(rel_angle, angle_bin)
            parts.append(f"{int(dist_q / 10)}:{int(angle_q / 10)}")

    label = "|".join(parts)
    return label