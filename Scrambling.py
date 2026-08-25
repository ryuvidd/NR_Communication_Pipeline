import numpy as np

class PDSCHScrambler():

    def __compute_c_init__(self, n_RNTI:int, q:int, n_ID:int) -> int:
        return n_RNTI * (2 ** 15) + q * (2 ** 14) + n_ID

    def __init__(self, n_RNTI:int, q:int, n_ID:int):
        self.c_init = self.__compute_c_init__(n_RNTI, q, n_ID)
    
    def process(self, Codeword: np.ndarray) -> np.ndarray:
        sequenceLength = len(Codeword)
        x1 = np.zeros(sequenceLength + 1600 + 31, dtype=np.uint8)
        x2 = np.zeros(sequenceLength + 1600 + 31, dtype=np.uint8)

        x1[0] = 1
        for i in range(31):
            x2[i] = (self.c_init >> i) & 1

        for n in range(sequenceLength + 1600):
            x1[n + 31] = x1[n + 3] ^ x1[n]
            x2[n + 31] = (x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n])

        c = x1[1600:1600 + sequenceLength] ^ x2[1600:1600 + sequenceLength]

        scrambledBits = Codeword.astype(np.uint8) ^ c
        return scrambledBits
    
if __name__ == "__main__":
    Scrambler = PDSCHScrambler(0x1234, 0, 0)
    Codeword = np.array([0,0,1,1,0])
    ScrambledBits = Scrambler.process(Codeword)