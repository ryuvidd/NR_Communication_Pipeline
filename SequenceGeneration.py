import numpy as np

class SequenceGenerator():
    def process(self, sequenceLength, c_init) -> np.ndarray:
        x1 = np.zeros(sequenceLength + 1600 + 31, dtype=np.uint8)
        x2 = np.zeros(sequenceLength + 1600 + 31, dtype=np.uint8)

        x1[0] = 1
        for i in range(31):
            x2[i] = (c_init >> i) & 1

        for n in range(sequenceLength + 1600):
            x1[n + 31] = x1[n + 3] ^ x1[n]
            x2[n + 31] = (x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n])

        c = x1[1600:1600 + sequenceLength] ^ x2[1600:1600 + sequenceLength]

        return c
    
if __name__ == "__main__":
    n_RNTI = 99
    q = 0
    n_ID = 42
    seqlength = 1200
    c_init = n_RNTI * (2 ** 15) + q * (2 ** 14) + n_ID
    ThisSequenceGenerator = SequenceGenerator()
    Sequence = ThisSequenceGenerator.process(seqlength, c_init)