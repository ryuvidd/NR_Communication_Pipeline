import numpy as np

class ZeroForcingEqualizer():
    def __init__(self, allocatedPRB, allocatedPDSCHSymbols) -> None:
        start_k = allocatedPRB[0] * 12
        last_k = allocatedPRB[-1] * 12 + 11
        self.allocated_k = [k for k in range(start_k,last_k+1)]
        self.allocated_l = allocatedPDSCHSymbols

    def process(self, EstimatedChannel: np.ndarray, EstimatedGrid: np.ndarray):
        EqualizedGrid = np.zeros_like(EstimatedChannel, dtype=np.complex128)
        k_idx = np.asarray(self.allocated_k)
        l_idx = np.asarray(self.allocated_l)
        EstimatedChannel[np.ix_(k_idx, l_idx)] += 1e-12
        EqualizedGrid[np.ix_(k_idx, l_idx)] = (EstimatedGrid[np.ix_(k_idx, l_idx)] / EstimatedChannel[np.ix_(k_idx, l_idx)])
        return EqualizedGrid

        
