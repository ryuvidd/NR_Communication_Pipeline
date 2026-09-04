import numpy as np
import math
from dataclasses import dataclass

@dataclass
class RateMatchingConfig:
    nLayer: int
    Qm: int
    baseGraph: int
    Z_c: int
    G: int

class RateMatcher():
    def __init__(self, config: RateMatchingConfig):
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

    def bitSelection(self, encodedCodeBlocks: list[np.ndarray], rv_id:int) -> list[np.ndarray]:
        C = len(encodedCodeBlocks)
        self.E = self.__calculate_E__(C)

        SelectedBits = []
        for r, E_ in enumerate(self.E):
            N = len(encodedCodeBlocks[r])
            N_cb = N
            k_0 = self.__select_k0__(rv_id, N_cb)
            k = 0
            j = 0
            selectedBit = np.zeros(E_, dtype=np.uint8)
            while k < E_:
                idx = (k_0 + j) % N_cb
                if encodedCodeBlocks[r][idx] != -1:
                    selectedBit[k] = encodedCodeBlocks[r][idx]
                    k += 1
                j += 1
            SelectedBits.append(selectedBit)
        return SelectedBits
    
    def bitInterleave(self, selectedBits: list[np.ndarray]) -> list[np.ndarray]:
        InterleavedCodeBlocks = []
        for r, E_ in enumerate(self.E):
            e = selectedBits[r]
            f = np.zeros(E_, dtype=np.uint8)
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
    
class RateRecoverer():
    def __init__(self, config: RateMatchingConfig, C: int, fillerMask: list[np.ndarray]):
        self.nLayer = config.nLayer
        self.Qm = config.Qm
        self.baseGraph = config.baseGraph
        self.Z_c = config.Z_c
        self.G = config.G
        self.E = self.__calculate_E__(C)
        self.N_cb = self.__compute__Ncb__(config.baseGraph, config.Z_c)
        self.fillerMask = []
        for i in range(C):
            if len(fillerMask) != C: 
                raise ValueError("Number of code blocks does not match.")
            filler = np.zeros(self.N_cb, dtype=bool)
            filler[:len(fillerMask[i]) - (2 * self.Z_c)] = fillerMask[i][2 * config.Z_c:]
            self.fillerMask.append(filler)

    def __select_k0__(self, rv_id: int, N_cb: int):

        if rv_id == 0: 
            return 0
        if self.baseGraph == 1:
            if rv_id == 1:
                return math.floor(17 * N_cb / (66 * self.Z_c)) * self.Z_c
            elif rv_id == 2:
                return math.floor(33 * N_cb / (66 * self.Z_c)) * self.Z_c
            elif rv_id == 3:
                return math.floor(56 * N_cb / (66 * self.Z_c)) * self.Z_c
        elif self.baseGraph == 2:
            if rv_id == 1:
                return math.floor(13 * N_cb / (50 * self.Z_c)) * self.Z_c
            elif rv_id == 2:
                return math.floor(25 * N_cb / (50 * self.Z_c)) * self.Z_c
            elif rv_id == 3:
                return math.floor(43 * N_cb / (50 * self.Z_c)) * self.Z_c
        raise ValueError("Invalid rv_id")

    def __calculate_E__(self, C: int):

        A = self.nLayer * self.Qm
        base = self.G // (A * C)
        remainder = (self.G // A) % C

        E = []
        for r in range(C):
            if r < C - remainder: E.append(A * base)
            else: E.append(A * (base + 1))
        if sum(E) != self.G:
            raise ValueError("Calculated E values do not sum to G.")
        return E
    
    def __compute__Ncb__(self, basegraph, Zc):
        if basegraph not in (1,2):
            raise ValueError("basegraph must be 1 or 2.")
        if basegraph == 1: N = 66 * Zc
        else: N = 50 * Zc
        return N
    
    def bitDeinterleave(self, DescrambledLLRs: list[np.ndarray]) -> list[np.ndarray]:
        DeinterleavedLLRs = []
        for r, E_ in enumerate(self.E):
            f = DescrambledLLRs[r]
            if len(f) != E_:
                raise ValueError(f"Expected {E_} LLRs for code block {r}, but received {len(f)}.")

            e = np.zeros(E_, dtype=np.float64)
            for j in range(E_ // self.Qm):
                for i in range(self.Qm):
                    source_idx = i + j * self.Qm
                    destination_idx = (i * E_) // self.Qm + j
                    e[destination_idx] = f[source_idx]

            DeinterleavedLLRs.append(e)
        return DeinterleavedLLRs
    
    def inverseBitSelection(self, DeinterleavedLLRs: list[np.ndarray], rv_id: int) -> list[np.ndarray]:
        RecoveredLLRs = []
        for r, e in enumerate(DeinterleavedLLRs):
            k_0 = self.__select_k0__(rv_id, self.N_cb)
            recovered = np.zeros(self.N_cb, dtype=np.float64)
            k = 0
            j = 0
            while k < len(e):
                idx = (k_0 + j) % self.N_cb
                if not self.fillerMask[r][idx]:
                    recovered[idx] += e[k]
                    k += 1
                j += 1
            RecoveredLLRs.append(recovered)
        return RecoveredLLRs
    
    def process(self, DescrambledLLRs: np.ndarray, rv_id: int) -> list[np.ndarray]:
        # Code block segmentation
        DescrambledLLRsCodeBlocks = []
        start_idx = 0
        for E_ in self.E:
            DescrambledLLRsCodeBlocks.append(DescrambledLLRs[start_idx:start_idx+E_])
            start_idx += E_

        # Rate Recovery
        DeinterleavedBlock = self.bitDeinterleave(DescrambledLLRsCodeBlocks)
        EstimatedEncodedLLRsCodeBlocks = self.inverseBitSelection(DeinterleavedBlock, rv_id)
        return EstimatedEncodedLLRsCodeBlocks

if __name__ == "__main__":
    config = RateMatchingConfig(
        nLayer=1,
        Qm=4,
        baseGraph=1,
        Z_c=320,
        G=26412
    )

    C = 3
    N_cb = 66 * config.Z_c
    rv_id = 1

    rng = np.random.default_rng(12345)
    EncodedCodeBlocks = []
    fillerMasks = []
    for r in range(C):
        # The matcher uses the punctured N_cb-bit buffer. The recoverer gets
        # the full-codeword mask, including the 2*Z_c punctured positions.
        fillerMask = np.zeros(N_cb + 2 * config.Z_c, dtype=bool)
        filler_start = 2 * config.Z_c + 1000 + 20 * r
        fillerMask[filler_start:filler_start + 20] = True
        encodedBlock = rng.integers(0, 2, size=N_cb, dtype=np.int8)
        encodedBlock[fillerMask[2 * config.Z_c:]] = -1

        EncodedCodeBlocks.append(encodedBlock)
        fillerMasks.append(fillerMask)

    rateMatcher = RateMatcher(config)
    rateMatchedCodeBlocks = rateMatcher.process(EncodedCodeBlocks, rv_id)
    TransmittedLLRs = [np.where(block == 0, 2.0, -2.0) for block in rateMatchedCodeBlocks]

    print(f"C       = {C}")
    print(f"N_cb    = {N_cb}")
    print(f"E       = {rateMatcher.E}")
    print(f"G       = {config.G}")
    print(f"RV      = {rv_id}")
    print()

    total_errors = 0

    for r in range(C):
        original = EncodedCodeBlocks[r]

        # Test this transmitted LLR block on its own. The one-block G value
        # is exactly E[r], so this call does not concatenate or split blocks.
        block_config = RateMatchingConfig(
            nLayer=config.nLayer,
            Qm=config.Qm,
            baseGraph=config.baseGraph,
            Z_c=config.Z_c,
            G=rateMatcher.E[r]
        )
        block_recoverer = RateRecoverer(block_config, C=1, fillerMask=[fillerMasks[r]])
        recovered = block_recoverer.process(TransmittedLLRs[r], rv_id)[0]
        fillerMask = block_recoverer.fillerMask[0]

        # Determine which circular-buffer positions should have been transmitted.
        transmittedMask = np.zeros(N_cb, dtype=bool)

        k0 = block_recoverer.__select_k0__(rv_id, N_cb)
        k = 0
        j = 0
        while k < rateMatcher.E[r]:
            idx = (k0 + j) % N_cb
            if not fillerMask[idx]:
                transmittedMask[idx] = True
                k += 1
            j += 1
        # Expected LLRs
        expected = np.zeros(N_cb, dtype=np.float64)
        valid = transmittedMask & (~fillerMask)
        expected[valid] = np.where(original[valid] == 0, 2.0, -2.0)

        # Compare
        errorMask = ~np.isclose(recovered, expected)
        numErrors = np.count_nonzero(errorMask)
        total_errors += numErrors

        print(f"Code block {r}:")
        print(f"  transmitted positions : {np.count_nonzero(transmittedMask)}")
        print(f"  expected positions    : {np.count_nonzero(valid)}")
        print(f"  recovery errors       : {numErrors}")

        if numErrors == 0:
            print("  PASS")
        else:
            print("  FAIL")
            errorIndices = np.where(errorMask)[0][:10]
            print("  First errors:")
            for idx in errorIndices:
                print(
                    f"    idx={idx}, "
                    f"original={original[idx]}, "
                    f"expected={expected[idx]}, "
                    f"recovered={recovered[idx]}"
                )
        print()

    if total_errors == 0:
        print("===================================")
        print("RATE RECOVERY TEST PASSED")
        print("===================================")
    else:
        print("===================================")
        print(f"RATE RECOVERY TEST FAILED: {total_errors} errors")
        print("===================================")
