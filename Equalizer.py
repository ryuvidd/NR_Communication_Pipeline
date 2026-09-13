import numpy as np

class ZeroForcingEqualizer():
    def __init__(self, allocatedPRB, allocatedPDSCHSymbols) -> None:
        start_k = allocatedPRB[0] * 12
        last_k = allocatedPRB[-1] * 12 + 11
        self.allocated_k = [k for k in range(start_k,last_k+1)]
        self.allocated_l = allocatedPDSCHSymbols

    def process(self, EstimatedChannel: np.ndarray, EstimatedGrid: np.ndarray, N0: float) -> tuple[np.ndarray, np.ndarray]:
        eps = 1e-12
        EqualizedGrid = np.zeros_like(EstimatedChannel, dtype=np.complex128)
        EffectiveNoiseVariance = np.zeros_like(EstimatedChannel, dtype=float)
        for k in self.allocated_k:
            for l in self.allocated_l:
                H = EstimatedChannel[k, l]
                Y = EstimatedGrid[k, l]

                if np.abs(H) < eps:
                    EqualizedGrid[k,l] = 0
                    EffectiveNoiseVariance[k,l] = np.inf
                else:
                    EqualizedGrid[k,l] = Y / H
                    EffectiveNoiseVariance[k,l] = N0 / np.abs(H)**2
        return EqualizedGrid, EffectiveNoiseVariance

        
