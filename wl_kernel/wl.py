import math
import networkx as nx
import hashlib
import numpy as np

class WLkernel:

    def __init__(self, Graph, k, size_vector, dist_quan = 10, angle_quan = 10):
        self.G = Graph
        self.k = k
        self.size = size_vector
        self.dist_quan = dist_quan
        self.angle_quan = angle_quan

    def stable_hash(self, label):
        """
        Hash string label u indeks 0..(size-1) pomoću md5
        """
        h = hashlib.md5(label.encode('utf-8')).hexdigest()
        return int(h, 16) % self.size
    
    def stable_hash2(self, label):
        return int(hashlib.sha256(label.encode()).hexdigest(), 16) % self.size


    def graph_to_frequency_vector(self):
        vec = np.zeros(self.size, dtype=np.uint8)

        for v in self.G.nodes:
            label = self.get_spatial_label(v)
            if label == "empty" or label == "NA":
                continue
            idx = self.stable_hash(label)
            vec[idx] += 1

        return vec

    def get_spatial_label(self, v):
        if 'x' not in self.G.nodes[v] or 'y' not in self.G.nodes[v]:
            return "NA"

        x0, y0 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
        pos_v = (x0, y0)

        neighbors = []
        for u in self.G.neighbors(v):
            if 'x' not in self.G.nodes[u] or 'y' not in self.G.nodes[u]:
                continue
            x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
            pos_u = (x1, y1)
            dist = euclidean(pos_v, pos_u)
            angle = angle_between(pos_v, pos_u)
            neighbors.append((u, dist, angle))

        # Sort by score descending
        neighbors.sort(key=lambda tup: tup[1])
        top_k = neighbors[:self.k]

        if not top_k:
            return "empty"

        # Bazni ugao = ugao ka prvom susjedu
        base_angle = top_k[0][2]

        parts = []
        for i, (u, dist, angle) in enumerate(top_k):
            rel_angle = (angle - base_angle) % 360
            dist_q = quantize(dist, self.dist_quan)
            
            if i == 0:
                parts.append(str(dist_q))  # bez ugla
            else:
                angle_q = quantize(rel_angle, self.angle_quan)
                parts.append(f"{dist_q}:{angle_q}")

        label = "|".join(parts)
        return label
    
    # def graph_to_binary_vector(self):
    #     vec = np.zeros(self.size, dtype=np.uint8)

    #     for v in self.G.nodes:
    #         label = self.get_spatial_label(v)
    #         if label == "empty" or label == "NA":
    #             continue
    #         idx = self.stable_hash(label)
    #         vec[idx] = 1

    #     return vec


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle_between(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx)) % 360

def quantize(value, base):
    return int(value / base)
