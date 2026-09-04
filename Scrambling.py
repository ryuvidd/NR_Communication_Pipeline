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
        #!!!! FOR SUPPORTING 2 CODEWORDS, Codeword must be list[np.ndarray] !!!!#
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
        
class PDSCHDescrambler():
    def __init__(self, n_RNTI:int, n_Codeword:int, n_ID:int):
        self.n_RNTI = n_RNTI
        self.n_Codeword = n_Codeword
        self.n_ID = n_ID
        self.SequenceGenerator = SequenceGenerator()

    def __compute_c_init__(self, n_RNTI:int, q:int, n_ID:int) -> int:
        return n_RNTI * (2 ** 15) + q * (2 ** 14) + n_ID
    
    def process(self, ScrambledLLRs: list[np.ndarray]) -> np.ndarray:
        #!!!! FOR SUPPORTING 2 CODEWORDS, this function must return list[np.ndarray] !!!!
        if len(ScrambledLLRs) != self.n_Codeword:
            raise ValueError(f"Expected {self.n_Codeword} codewords.")

        if self.n_Codeword == 1:
            c_init = self.__compute_c_init__(self.n_RNTI, 0, self.n_ID)
            sequenceLength = len(ScrambledLLRs[0])
            c = self.SequenceGenerator.process(sequenceLength, c_init)
            signs = 1 - 2 * c.astype(np.int8)
            DescrambledLLRs = signs * ScrambledLLRs[0]
        return DescrambledLLRs
    
if __name__ == "__main__":
    n_RNTI = 0x1234
    q = 1
    n_ID = 0
    NumBits = 100000

    Scrambler = PDSCHScrambler(n_RNTI, q, n_ID)
    Codeword = np.random.randint(0, 2, NumBits)
    ScrambledBits = Scrambler.process(Codeword)

    print("Assume Ideal Mapper -> LLR is either 2 (0) or -2 (1)")
    ScrambledLLRs = np.array([-2 if a == 1 else 2 for a in ScrambledBits[0]], dtype=np.float64)
    Descrambler = PDSCHDescrambler(n_RNTI, q, n_ID)
    DescrambledLLRs = Descrambler.process([ScrambledLLRs])
    DescrambledBits = np.array([1 if a == -2 else 0 for a in DescrambledLLRs[0]], dtype=np.uint8)
    
    print("Maximum absolute difference: ", np.max(np.abs(Codeword - DescrambledBits)))
    print("No significant error: ", np.allclose(Codeword, DescrambledBits))