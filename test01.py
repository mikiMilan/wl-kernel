import numpy as np

class SimHasher:
    def __init__(self, dim, n_bits=128, seed=None):
        self.dim = dim
        self.n_bits = n_bits
        rng = np.random.default_rng(seed)
        self.random_vectors = rng.standard_normal((n_bits, dim))

    def hash(self, vec):
        vec = np.array(vec)
        projections = self.random_vectors @ vec  # skalarni proizvodi
        return ''.join(['1' if val > 0 else '0' for val in projections])
    
        # ili
        vec = np.array(vec)
        projections = self.random_vectors @ vec
        ba = bitarray([val > 0 for val in projections])
        return int(ba.to01(), 2)

    def hamming_distance(self, h1, h2):
        return sum(c1 != c2 for c1, c2 in zip(h1, h2))



hasher = SimHasher(dim=5, n_bits=32, seed=42)

v1 = (4, 5, 7, 9, 12)
# v1 = (4, 5, 10, 9, 12)
v2 = (4, 5, 6, 9, 12)

h1 = hasher.hash(v1)
h2 = hasher.hash(v2)

print("SimHash v1:", h1)
print("SimHash v2:", h2)
print("Hamming distanca:", hasher.hamming_distance(h1, h2))