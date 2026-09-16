import numpy as np
from dataclasses import dataclass
from enum import Enum
from Configuration import *

@dataclass
class EqualizerConfig:
    allocatedPRB: list
    allocatedPDSCHSymbols: list

def select_equalizer(equalizer_type: EQUALIZER, config: EqualizerConfig):
    if equalizer_type == EQUALIZER.ZF:
        return ZeroForcingEqualizer(config)
    elif equalizer_type == EQUALIZER.MMSE:
        return MMSE_Equalizer(config)
    else:
        raise ValueError("Only support Zero-Forcing and MMSE Equalizers for now")
    
class MMSE_Equalizer():
    def __init__(self, config: EqualizerConfig):
        start_k = config.allocatedPRB[0] * 12
        last_k = config.allocatedPRB[-1] * 12 + 11
        self.allocated_k = [k for k in range(start_k,last_k+1)]
        self.allocated_l = config.allocatedPDSCHSymbols

    def process(self, EstimatedChannel: np.ndarray, EstimatedGrid: np.ndarray, meta:dict) -> tuple[np.ndarray, dict]:
        N0 = meta["NoiseVar"]
        Es = 1
        EqualizedGrid = np.zeros_like(EstimatedChannel, dtype=np.complex128)
        EffectiveVarNoise = np.zeros_like(EstimatedChannel, dtype=float)
        gain = np.zeros_like(EstimatedChannel, dtype=float)
        for k in self.allocated_k:
            for l in self.allocated_l:
                H = EstimatedChannel[k, l]
                Y = EstimatedGrid[k, l]

                W_mmse = H.conj() / (np.abs(H)**2 + (N0 / Es))
                a = np.real(W_mmse * H)

                EqualizedGrid[k,l] = W_mmse * Y
                EffectiveVarNoise[k,l] = np.abs(W_mmse) ** 2 * N0
                gain[k,l] = a
        
        meta["EffectiveVarNoise"] = EffectiveVarNoise
        meta["gain"] = gain
        return EqualizedGrid, meta

class ZeroForcingEqualizer():
    def __init__(self, config:EqualizerConfig):
        start_k = config.allocatedPRB[0] * 12
        last_k = config.allocatedPRB[-1] * 12 + 11
        self.allocated_k = [k for k in range(start_k,last_k+1)]
        self.allocated_l = config.allocatedPDSCHSymbols

    def process(self, EstimatedChannel: np.ndarray, EstimatedGrid: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
        eps = 1e-12
        EqualizedGrid = np.zeros_like(EstimatedChannel, dtype=np.complex128)
        EffectiveVarNoise = np.zeros_like(EstimatedChannel, dtype=float)
        gain = np.zeros_like(EstimatedChannel, dtype=float)
        for k in self.allocated_k:
            for l in self.allocated_l:
                H = EstimatedChannel[k, l]
                Y = EstimatedGrid[k, l]

                gain[k,l] = 1

                if np.abs(H) < eps:
                    EqualizedGrid[k,l] = 0
                    EffectiveVarNoise[k,l] = np.inf
                else:
                    EqualizedGrid[k,l] = Y / H
                    EffectiveVarNoise[k,l] = meta["VarNoise"] / np.abs(H)**2

        meta["EffectiveVarNoise"] = EffectiveVarNoise
        meta["gain"] = gain
        return EqualizedGrid, meta

        
