import numpy as np
from dataclasses import dataclass
from DMRSGeneration import *

@dataclass
class ResourceMappingConfig:
    nPRB: int
    allocatedPRB: list
    allocatedPDSCHSymbols: list
    allocatedDMRSPerPRB: list[tuple]
    nLayer: int
    nOFDMSymbolsPerSlot: int
    slotNumInFrame: int
    N_DMRS_ID: int
    lambda_bar: int
    n_SCID: int

class ResourceMapper():
    def __init__(self, config:ResourceMappingConfig):
        self.nPRB = config.nPRB
        self.allocatedPRB = config.allocatedPRB
        self.allocatedDMRSPerPRB = config.allocatedDMRSPerPRB
        self.allocatedPDSCHSymbols = config.allocatedPDSCHSymbols
        self.nLayer = config.nLayer
        self.nOFDMSymbolsPerSlot = config.nOFDMSymbolsPerSlot
        self.RETypeGrid = self.__get_RE_Type_Grid__()

    def __get_RE_Type_Grid__(self) -> np.ndarray:
        RETypegrid = np.full((self.nPRB * 12, self.nOFDMSymbolsPerSlot), "UNUSED", dtype=object)

        for prb in self.allocatedPRB:
            start_k = prb * 12
            for l in self.allocatedPDSCHSymbols:
                RETypegrid[start_k:start_k + 12, l] = "PDSCH_DATA"

        for prb in self.allocatedPRB:
            start_k = prb * 12
            for k_offset, l in self.allocatedDMRSPerPRB:
                RETypegrid[start_k + k_offset, l] = "DMRS"
        
        return RETypegrid
    
    def process(self, LayerMappedSymbols: list[np.ndarray], DMRSs: dict) -> np.ndarray:
        if self.nLayer != 1:
            raise NotImplementedError("Only single-layer transmission is currently supported.")
        if len(LayerMappedSymbols) != 1:
            raise ValueError("Expected exactly one layer.")
        if self.nLayer != 1:
            raise NotImplementedError("Only single-layer transmission is currently supported.")
        if len(LayerMappedSymbols) != 1:
            raise ValueError("Expected exactly one layer.")
        numExpectedData = np.count_nonzero(self.RETypeGrid == "PDSCH_DATA")
        numExpectedDMRS = np.count_nonzero(self.RETypeGrid == "DMRS")
        if len(LayerMappedSymbols[0]) != numExpectedData:
            raise ValueError(f"Expected {numExpectedData} PDSCH symbols, but received {len(LayerMappedSymbols[0])}.")
        totalDMRS = sum(len(dmrs) for dmrs in DMRSs.values())
        if totalDMRS != numExpectedDMRS:
            raise ValueError(f"Expected {numExpectedDMRS} DM-RS symbols, but received {totalDMRS}.")
        
        grid = np.zeros((self.nPRB * 12, self.nOFDMSymbolsPerSlot), dtype=np.complex64)
        data_idx = 0
        for l in self.allocatedPDSCHSymbols:
            dmrs_idx = 0
            for prb in self.allocatedPRB:
                start_k = prb * 12
                for k in range(start_k, start_k + 12):
                    if self.RETypeGrid[k, l] == "DMRS":
                        grid[k, l] = DMRSs[l][dmrs_idx]
                        dmrs_idx += 1
                    elif self.RETypeGrid[k, l] == "PDSCH_DATA":
                        grid[k, l] = LayerMappedSymbols[0][data_idx]
                        data_idx += 1
        return grid
    
if __name__ == "__main__":
    config = ResourceMappingConfig(
        nPRB = 50,
        allocatedPRB = [a for a in range(5, 15)],
        allocatedPDSCHSymbols = [a for a in range(2,14)],
        allocatedDMRSPerPRB = [(0,2),(2,2),(4,2),(6,2),(8,2),(10,2)],
        nLayer = 1,
        nOFDMSymbolsPerSlot = 14,
        slotNumInFrame = 0,
        N_DMRS_ID = 100,
        lambda_bar = 0,
        n_SCID = 0
    )

    numAllocatedPRB = len(config.allocatedPRB)
    numPDSCHSymbols = len(config.allocatedPDSCHSymbols)
    numDMRS = len(config.allocatedDMRSPerPRB)
    numExpectedDataSymbols = numAllocatedPRB * config.nLayer * (numPDSCHSymbols * 12 - numDMRS)
    numExpectedDMRS = numAllocatedPRB * config.nLayer * numDMRS
    LayerMappedSymbols = [np.arange(numExpectedDataSymbols)+50]
    DMRSs = {'2': np.arange(numExpectedDMRS)+2}

    REMapper = ResourceMapper(config)
    grid = REMapper.process(LayerMappedSymbols, DMRSs)