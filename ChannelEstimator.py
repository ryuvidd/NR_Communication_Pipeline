import numpy as np
from dataclasses import dataclass

@dataclass
class ChannelEstimatorConfig:
    allocatedPRB: list
    allocatedPDSCHSymbols: list
    allocatedDMRSPerPRB: list[tuple]
    RETypeGrid: np.ndarray


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

    def process(self, EstimatedGrid: np.ndarray, DMRSs: dict) -> np.ndarray:
        EstimatedDMRSChannelGrid = self.__estimate_channels__(EstimatedGrid, DMRSs)
        EstimatedChannels = self.__interpolate_channels__(EstimatedDMRSChannelGrid, DMRSs)
        return EstimatedChannels
