import numpy as np
import math
from dataclasses import dataclass

@dataclass
class RateMatchConfig:
    nLayer: int
    Qm: int
    baseGraph: int
    Z_c: int
    G: int

class RateMatcher():
    def __init__(self, config: RateMatchConfig):
        self.nLayer = config.nLayer
        self.Qm = config.Qm
        self.baseGraph = config.baseGraph
        self.Z_c = config.Z_c
        self.G = config.G
        
    def __select_k0__(self, rv_id, N_cb):
        if rv_id == 0:
            return 0
        else:
            if self.baseGraph == 1:
                if rv_id == 1: return math.floor(17 * N_cb / 66 / self.Z_c) * self.Z_c
                elif rv_id == 2: return math.floor(33 * N_cb / 66 / self.Z_c) * self.Z_c
                else: return math.floor(56 * N_cb / 66 / self.Z_c) * self.Z_c
            else:
                if rv_id == 1: return math.floor(13 * N_cb / 50 / self.Z_c) * self.Z_c
                elif rv_id == 2: return math.floor(25 * N_cb / 50 / self.Z_c) * self.Z_c
                else: return math.floor(43 * N_cb / 50 / self.Z_c) * self.Z_c

    def __calculate_E__(self, C: int):
        A = self.nLayer * self.Qm

        base = self.G // (A * C)
        remainder = (self.G // A)  % C

        E = []
        for r in range(C):
            if r < C - remainder: E.append(A * base)
            else: E.append(A * (base + 1))
        assert sum(E) == self.G
        return E

    def bitSelection(self, encodedCodeBlocks: list, rv_id:int) -> list:
        C = len(encodedCodeBlocks)
        self.E = self.__calculate_E__(C)

        SelectedBits = []
        for r, E_ in enumerate(self.E):
            N = len(encodedCodeBlocks[r])
            N_cb = N
            k_0 = self.__select_k0__(rv_id, N_cb)
            k = 0
            j = 0
            selectedBit = np.zeros(E_, dtype=np.int8)
            while k < E_:
                idx = (k_0 + j) % N_cb
                if encodedCodeBlocks[r][idx] != -1:
                    selectedBit[k] = encodedCodeBlocks[r][idx]
                    k += 1
                j += 1
            SelectedBits.append(selectedBit)
        return SelectedBits
    
    def bitInterleave(self, selectedBits: list) -> list:
        InterleavedCodeBlocks = []
        for r, E_ in enumerate(self.E):
            e = selectedBits[r]
            f = np.zeros(E_, dtype=np.int8)
            for j in range(E_ // self.Qm):
                for i in range(self.Qm):
                    f[i+j*self.Qm] = e[(i*E_)//self.Qm + j]
            InterleavedCodeBlocks.append(f)
        return InterleavedCodeBlocks
    
    def process(self, encodedCodeBlocks: list, rv_id:int) -> list:
        ## Following TS38.212 clause 5.4.2: rate matching for LDPC ##
        Selectedbits = self.bitSelection(encodedCodeBlocks, rv_id)
        InterleavedCodeBlocks = self.bitInterleave(Selectedbits)
        return InterleavedCodeBlocks
    
if __name__ == "__main__":
    config = RateMatchConfig(
        nLayer = 1,
        Qm = 4,
        baseGraph = 1,
        Z_c = 320,
        G = 26400
    )
    C = 2
    N = 21120
    rng = np.random.default_rng()
    EncodedCodeBlocks = []
    for r in range(C):
        encodedBlock = rng.integers(0, 2, size=N, dtype=np.int8)
        EncodedCodeBlocks.append(encodedBlock)
    ThisRateMatcher = RateMatcher(config)
    InterleavedCodeBlocks = ThisRateMatcher.process(EncodedCodeBlocks, rv_id=1)
    
