import numpy as np
from dataclasses import dataclass
from Configuration import *

@dataclass
class ChannelEstimatorConfig:
    allocatedPRB: list
    allocatedPDSCHSymbols: list
    allocatedDMRSPerPRB: list[tuple]
    RETypeGrid: np.ndarray

def select_estimator(estimator_type: CHANNEL_ESTIMATOR, config: ChannelEstimatorConfig):
    if estimator_type == CHANNEL_ESTIMATOR.LS:
        return LSEstimator(config)
    elif estimator_type == CHANNEL_ESTIMATOR.LMMSE:
        return LMMSEEstimator(config)
    else:
        raise ValueError("Only support LS and LMMSE Estimators for now")

class LMMSEEstimator():
    def __init__(self, config: ChannelEstimatorConfig):
        self.allocatedPDSCHSymbols = config.allocatedPDSCHSymbols
        allocatedSubcarriers = []
        for prb in config.allocatedPRB:
            start_sc = prb * 12
            end_sc = start_sc + 12
            allocatedSubcarriers.extend(range(start_sc, end_sc))
        self.allocatedSubcarriers = allocatedSubcarriers
        self.allocatedDMRSPerPRB = config.allocatedDMRSPerPRB
        self.RETypeGrid = config.RETypeGrid
        self.DMRSIndices = self.__compute_dmrs_k__(config.allocatedPRB, config.allocatedDMRSPerPRB)

    def __compute_dmrs_k__(self, allocatedPRB, allocatedDMRSPerPRB) -> dict:
        l_positions = {}
        for (k,l) in allocatedDMRSPerPRB:
            if l not in l_positions.keys():
                l_positions[l] = [k]
            else:
                l_positions[l].append(k)
        
        DMRS_positions = {}
        for l in l_positions:
            DMRS_positions[l] = []
            for prb in allocatedPRB:
                k_offset = prb * 12
                DMRS_positions[l].append(np.array(l_positions[l]) + k_offset)
            DMRS_positions[l] = np.concatenate(DMRS_positions[l])
        return DMRS_positions
            
    def __estimate_channels__(self, EstimatedGrid: np.ndarray, DMRSs: dict, meta: dict) -> np.ndarray:
        mean_h = meta["mean_h"]
        R_hh = meta["R_hh"]
        NoiseVar = meta["NoiseVar"]

        nActiveSubcarrier = EstimatedGrid.shape[0]
        NFFT = mean_h.shape[0]
        start_k = NFFT // 2 - nActiveSubcarrier // 2
        last_k = start_k + nActiveSubcarrier
        mean_h = mean_h[start_k:last_k][self.allocatedSubcarriers]
        R_hh = R_hh[start_k:last_k, start_k:last_k][self.allocatedSubcarriers][:,self.allocatedSubcarriers]

        EstimatedDMRSsChannel = np.zeros_like(EstimatedGrid)
        for l, dmrs in DMRSs.items():
            dmrs_idx = 0
            Y = np.zeros_like(dmrs, dtype=np.complex128)
            X = np.zeros((dmrs.size, len(self.allocatedSubcarriers)), dtype=np.complex128)
            for i,k in enumerate(self.allocatedSubcarriers):
                REType = self.RETypeGrid[k,l]
                if REType == "DMRS":
                    Y[dmrs_idx] = EstimatedGrid[k,l]
                    X[dmrs_idx,i] = dmrs[dmrs_idx]
                    dmrs_idx += 1

            EstimatedDMRSsChannel[self.allocatedSubcarriers, l] = (
                mean_h 
                + R_hh @ X.conj().T @ np.linalg.inv(X @ R_hh @ X.conj().T + NoiseVar * np.eye(dmrs.size)) @ (Y - X @ mean_h)
            )
        
        return EstimatedDMRSsChannel
    
    def __interpolate_channels__(self, EstimatedDMRSsChannel: np.ndarray, DMRSs: dict) -> np.ndarray:
        EstimatedChannels = EstimatedDMRSsChannel.copy()
        nSymbols = EstimatedChannels.shape[1]

        # Time Interpolate
        start_k = self.allocatedSubcarriers[0]
        last_k = self.allocatedSubcarriers[-1]
        l_positions = sorted(DMRSs.keys())  
        EstimatedChannels[start_k:last_k+1,:l_positions[0]] = EstimatedChannels[start_k:last_k+1,l_positions[0]][:,None]
        EstimatedChannels[start_k:last_k+1,l_positions[-1]+1:] = EstimatedChannels[start_k:last_k+1,l_positions[-1]][:, None]      
        if len(l_positions) > 1:
            l_interpolate = np.arange(l_positions[0], l_positions[-1]+1)
            for k in range(start_k, last_k+1):
                EstimatedChannels[k,l_positions[0]:l_positions[-1]+1] = np.interp(l_interpolate, l_positions, EstimatedChannels[k, l_positions])
        
        for l in range(nSymbols):
            if l not in self.allocatedPDSCHSymbols:
                EstimatedChannels[:,l] = 0
                
        return EstimatedChannels

    def process(self, EstimatedGrid: np.ndarray, DMRSs: dict, meta: dict) -> np.ndarray:
        EstimatedDMRSChannelGrid = self.__estimate_channels__(EstimatedGrid, DMRSs, meta)
        EstimatedChannels = self.__interpolate_channels__(EstimatedDMRSChannelGrid, DMRSs)
        return EstimatedChannels

class LSEstimator():
    def __init__(self, config: ChannelEstimatorConfig):
        self.allocatedPDSCHSymbols = config.allocatedPDSCHSymbols
        self.allocatedPRB = config.allocatedPRB
        self.allocatedDMRSPerPRB = config.allocatedDMRSPerPRB
        self.RETypeGrid = config.RETypeGrid
        self.DMRSIndices = self.__compute_dmrs_k__(config.allocatedPRB, config.allocatedDMRSPerPRB)

    def __compute_dmrs_k__(self, allocatedPRB, allocatedDMRSPerPRB) -> dict:
        l_positions = {}
        for (k,l) in allocatedDMRSPerPRB:
            if l not in l_positions.keys():
                l_positions[l] = [k]
            else:
                l_positions[l].append(k)
        
        DMRS_positions = {}
        for l in l_positions:
            DMRS_positions[l] = []
            for prb in allocatedPRB:
                k_offset = prb * 12
                DMRS_positions[l].append(np.array(l_positions[l]) + k_offset)
            DMRS_positions[l] = np.concatenate(DMRS_positions[l])
        return DMRS_positions
            
    def __estimate_channels__(self, EstimatedGrid: np.ndarray, DMRSs: dict) -> np.ndarray:
        EstimatedDMRSsChannel = np.zeros_like(EstimatedGrid)
        for l, dmrs in DMRSs.items():
            dmrs_idx = 0
            for prb in self.allocatedPRB:
                start_k = prb * 12
                for k in range(start_k, start_k + 12):
                    REType = self.RETypeGrid[k,l]
                    if REType == "DMRS":
                        EstimatedDMRSsChannel[k,l] = EstimatedGrid[k,l] / dmrs[dmrs_idx]
                        dmrs_idx += 1
        return EstimatedDMRSsChannel        

    
    def __interpolate_channels__(self, EstimatedDMRSsChannel: np.ndarray, DMRSs: dict) -> np.ndarray:
        EstimatedChannels = EstimatedDMRSsChannel.copy()
        nSymbols = EstimatedChannels.shape[1]

        # Frequency Interpolate
        start_k = self.allocatedPRB[0] * 12
        last_k = self.allocatedPRB[-1] * 12 + 11
        for l in DMRSs:
            k_dmrs = self.DMRSIndices[l]
            EstimatedChannels[start_k:k_dmrs[0]+1, l] = EstimatedDMRSsChannel[k_dmrs[0], l]
            EstimatedChannels[k_dmrs[-1]:last_k + 1, l] = EstimatedDMRSsChannel[k_dmrs[-1], l]
            k_interpolate = np.arange(k_dmrs[0], k_dmrs[-1]+1)
            estimated_interpolate = EstimatedDMRSsChannel[k_dmrs, l]
            EstimatedChannels[k_dmrs[0]:k_dmrs[-1]+1,l] = np.interp(k_interpolate, k_dmrs, estimated_interpolate)

        # Time Interpolate
        l_positions = sorted(DMRSs.keys())  
        EstimatedChannels[start_k:last_k+1,:l_positions[0]] = EstimatedChannels[start_k:last_k+1,l_positions[0]][:,None]
        EstimatedChannels[start_k:last_k+1,l_positions[-1]+1:] = EstimatedChannels[start_k:last_k+1,l_positions[-1]][:, None]      
        if len(l_positions) > 1:
            l_interpolate = np.arange(l_positions[0], l_positions[-1]+1)
            for k in range(start_k, last_k+1):
                EstimatedChannels[k,l_positions[0]:l_positions[-1]+1] = np.interp(l_interpolate, l_positions, EstimatedChannels[k, l_positions])
        
        for l in range(nSymbols):
            if l not in self.allocatedPDSCHSymbols:
                EstimatedChannels[:,l] = 0
                
        return EstimatedChannels

    def process(self, EstimatedGrid: np.ndarray, DMRSs: dict, meta: dict) -> np.ndarray:
        EstimatedDMRSChannelGrid = self.__estimate_channels__(EstimatedGrid, DMRSs)
        EstimatedChannels = self.__interpolate_channels__(EstimatedDMRSChannelGrid, DMRSs)
        return EstimatedChannels
