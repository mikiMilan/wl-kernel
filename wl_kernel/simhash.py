import numpy as np

class SimHashModulo:
    def __init__(self, bits=7, seed=42):
        self.bits = bits
        self.size = 1 << bits  # npr. 2^7 = 128 mogućih vrednosti
        self.random_vectors = {}
        self.rng = np.random.default_rng(seed)

    def _get_vector(self, feature: int) -> np.ndarray:
        if feature not in self.random_vectors:
            self.random_vectors[feature] = self.rng.choice([-1, 1], size=self.bits)
        return self.random_vectors[feature]

    def encode(self, s: str) -> int:
        # Parsira string npr. "4|1:3:6" → [4,1,3,6]
        features = list(map(int, s.replace('|', ':').split(':')))
        vec = np.zeros(self.bits)
        for f in features:
            vec += self._get_vector(f)
        bits = ['1' if x > 0 else '0' for x in vec]
        return int(''.join(bits), 2)  # broj u opsegu [0, 2^bits - 1]