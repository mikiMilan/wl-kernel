import xxhash
import numpy as np
from numpy.linalg import norm
import networkx as nx
from .utils import euclidean, angle_between, quantize
from functools import lru_cache

class WLkernel:
    """
    The WLkernel class generates vector representations of a graph
    based on various local node features, such as spatial position (x, y)
    or node degree.

    Attributes:
        graph (nx.Graph): Input graph with node attributes.
        size (int): Dimension of the resulting vector.
        k (int or None): Number of nearest neighbors used in spatial labeling.
    """
    def __init__(self, graph:nx.Graph, size:int = 128, k=None, ascending: bool = False):
        """
        Initializes the WLkernel object.

        Args:
            graph (nx.Graph): Input graph.
            size (int): Dimension of the output vector.
            k (int, optional): Number of nearest neighbors used for spatial labeling.
        """
        self.graph = graph
        self.k = k
        self.size = size # vector dimension
        self.ascending = ascending

    @lru_cache(maxsize=100000)
    def stable_hash(self, label) -> int:
        """
        Generates a stable hash for the given string label using xxhash.

        Args:
            label (str): String to be hashed.

        Returns:
            int: Index in the range [0, size).
        """
        return xxhash.xxh32(label).intdigest() % self.size

    def simhash(self, label) -> int:
        return xxhash.xxh32(label).intdigest() % self.size

    def spatial_vector(self, dist_quan=10, angle_quan=10) -> np.ndarray:
        """
        Generates a vector representing the frequency of spatial labels in the graph.

        A spatial label is constructed based on the k nearest neighbors of a node 
        (by distance), including quantized distance and relative angle.

        Args:
            dist_quan (int): Base value for distance quantization.
            angle_quan (int): Base value for angle quantization.

        Returns:
            np.ndarray: A vector of dimension `size` containing the frequencies of hashed labels.
        """
        vec = np.zeros(self.size, dtype=np.uint8)

        for v in self.graph.nodes:
            label = self.get_spatial_label(v, dist_quan, angle_quan)
            if label == "empty" or label == "NA":
                continue
            idx = self.stable_hash(label)
            vec[idx] += 1

        return vec
    
    def degree_vector(self, 
                      iter: int = 0, 
                      sort_edge_key: str = None, 
                      sort_node_key: str = None, 
                      degree_quan: int = 1) -> np.ndarray:
        """
        Generates a feature vector using the Weisfeiler-Lehman (WL) relabeling procedure.

        Each node is initially labeled using a base attribute (e.g., degree or a predefined label).
        In each iteration, the node's label is updated by concatenating its current label with the
        sorted multiset of its neighbors' labels. Sorting can be customized using edge or node
        attributes, or fallback to degree if none is provided.

        The final labels are counted to produce a vector representing label frequencies.

        Args:
            iter (int): Number of WL iterations to apply (0 means only initial labeling).
            sort_edge_key (str, optional): Edge attribute used to sort neighbors before relabeling.
            sort_node_key (str, optional): Node attribute used to sort neighbors before relabeling.
            k (int, optional): Maximum number of neighbors to consider when relabeling.
            ascending (bool, optional): Whether to sort in ascending order.

        Returns:
            np.ndarray: A vector of dimension `size` containing frequencies of final WL labels.
        """
        G = self.graph
        # 1. Pripremi top-k usmereni graf
        top_k_neighbors = self.get_top_k_neighbors(sort_edge_key, sort_node_key)
        
        # 2. Početne labele (string stepena iz originalnog grafa)
        labels = {v: str(G.degree[v]//degree_quan) for v in G.nodes}

        # 3. initial vector
        vec = np.zeros(self.size, dtype=np.uint32)
        for v in G.nodes:
            idx = self.stable_hash(labels[v])  # tuple hash
            vec[idx] += 1

        # 4. Iterativni WL relabeling
        for _ in range(iter):
            new_labels = {}
            for v in G.nodes:
                neighbors = top_k_neighbors[v]
                neighbor_labels = [str(labels[u]) for u in neighbors]
                new_labels[v] = str(labels[v]) + "|" + ":".join(neighbor_labels)
                new_labels[v] = self.stable_hash(new_labels[v])
                vec[new_labels[v]] += 1 
            labels = new_labels

        return vec

    def get_spatial_label(self, v, dist_quan, angle_quan):
        """
        Creates a spatial label for the given node `v` based on its nearest neighbors.

        The label includes quantized distances and relative angles to the neighbors.

        Args:
            v (int): ID of the node in the graph.
            dist_quan (int): Base value for distance quantization.
            angle_quan (int): Base value for angle quantization.

        Returns:
            str: Spatial label in the format 'd|d:a|d:a', or "empty"/"NA" if not applicable.
        """
        if 'x' not in self.graph.nodes[v] or 'y' not in self.graph.nodes[v]:
            return "NA"

        x0, y0 = self.graph.nodes[v]['x'], self.graph.nodes[v]['y']
        pos_v = (x0, y0)

        neighbors = []
        for u in self.graph.neighbors(v):
            if 'x' not in self.graph.nodes[u] or 'y' not in self.graph.nodes[u]:
                continue
            x1, y1 = self.graph.nodes[u]['x'], self.graph.nodes[u]['y']
            pos_u = (x1, y1)
            dist = euclidean(pos_v, pos_u)
            angle = angle_between(pos_v, pos_u)
            neighbors.append((u, dist, angle))

        # Sort by score descending
        neighbors.sort(key=lambda tup: tup[1], reverse=not self.ascending)
        top_k = neighbors[:self.k]

        if not top_k:
            return "empty"

        # Base angle = angle toward the first neighbor
        base_angle = top_k[0][2]

        parts = []
        for i, (u, dist, angle) in enumerate(top_k):
            rel_angle = (angle - base_angle) % 360
            dist_q = quantize(dist, dist_quan)
            
            if i == 0:
                parts.append(str(dist_q))  # without angle
            else:
                angle_q = quantize(rel_angle, angle_quan)
                parts.append(f"{dist_q}:{angle_q}")

        label = "|".join(parts)
        return label

    def get_top_k_neighbors(self, sort_edge_key=None, sort_node_key=None):
        """
        Returns a dictionary where each node is mapped to a list of its top-k neighbors,
        sorted according to the specified criterion.

        Neighbors can be sorted based on:
        - edge weight (if `sort_edge_key` is provided),
        - node attribute (if `sort_node_key` is provided),
        - or node degree (used as a fallback if no sorting key is given).

        The number of neighbors is limited to `k` if specified during class initialization.

        Args:
            sort_edge_key (str or None): Edge attribute used for sorting neighbors.
            sort_node_key (str or None): Node attribute used for sorting neighbors.

        Returns:
            dict: A mapping from each node to a list of its top-k sorted neighbors:
                {node: [neighbor1, neighbor2, ...]}
        """
        top_k_neighbors = {}
        G = self.graph

        for v in G.nodes:
            neighbors = list(G.successors(v)) if isinstance(G, nx.DiGraph) else list(G.neighbors(v))

            if sort_edge_key is not None:
                neighbors.sort(
                    key=lambda u: (G[v][u].get(sort_edge_key, 0), G.degree[u]),
                    reverse=not self.ascending
                )
            elif sort_node_key is not None:
                neighbors.sort(
                    key=lambda u: (G.nodes[u].get(sort_node_key, 0), G.degree[u]),
                    reverse=not self.ascending
                )
            else:
                neighbors.sort(
                    key=lambda u: G.degree[u],
                    reverse=not self.ascending
                )

            if self.k is not None:
                neighbors = neighbors[:self.k]

            top_k_neighbors[v] = neighbors

        return top_k_neighbors

    @staticmethod
    def similarity(vec1: np.ndarray, vec2: np.ndarray, method: str = "cosine") -> float:
        """
        Calculates similarity between two vectors using the specified method.

        Supported methods:
        - "cosine" (default)
        - "jaccard"

        Args:
            vec1 (np.ndarray): First vector.
            vec2 (np.ndarray): Second vector.
            method (str): Similarity metric to use.

        Returns:
            float: Similarity score (higher means more similar).
        """
        if method == "cosine":
            norm_product = norm(vec1) * norm(vec2)
            if norm_product == 0:
                return 0.0
            return np.dot(vec1, vec2) / norm_product

        elif method == "jaccard":
            intersection = np.minimum(vec1, vec2).sum()
            union = np.maximum(vec1, vec2).sum()
            if union == 0:
                return 0.0
            return intersection / union

        else:
            raise ValueError(f"Unsupported similarity method: {method}")
