import numpy as np
from SequenceGeneration import *

class PDSCHScrambler():

    def __init__(self, n_RNTI:int, n_Codeword:int, n_ID:int):
        self.n_RNTI = n_RNTI
        self.n_Codeword = n_Codeword
        self.n_ID = n_ID
        self.SequenceGenerator = SequenceGenerator()

    def __compute_c_init__(self, n_RNTI:int, q:int, n_ID:int) -> int:
        return n_RNTI * (2 ** 15) + q * (2 ** 14) + n_ID
    
    def process(self, Codeword: np.ndarray) -> list[np.ndarray]:
        
        if self.n_Codeword == 1:
            c_init = self.__compute_c_init__(self.n_RNTI, 0, self.n_ID)
            sequenceLength = len(Codeword)
            c = self.SequenceGenerator.process(sequenceLength, c_init)
            return [Codeword.astype(np.uint8) ^ c]
        else:
            # Codeword.shape = (2, seqlength) for this case
            scrambledBits = []
            for i in range(2):
                c_init = self.__compute_c_init__(self.n_RNTI, i, self.n_ID)
                sequenceLength = len(Codeword[i])
                c = self.SequenceGenerator.process(sequenceLength, c_init)
                scrambledBits.append(Codeword[i].astype(np.uint8) ^ c)
            return scrambledBits
    
if __name__ == "__main__":
    Scrambler = PDSCHScrambler(0x1234, 1, 0)
    Codeword = np.array([0,0,1,1,0])
    ScrambledBits = Scrambler.process(Codeword)