import numpy as np
from dataclasses import dataclass
from enum import Enum
from SequenceGeneration import *

class TRANSMISSION_TYPE(Enum):
    PDSCH = "PDSCH"
    PUSCH = "PUSCH"

@dataclass
class PDSCH_DMRS_GeneratorConfig:
    nOFDMSymbolsPerSlot: int
    allocatedDMRSPerPRB: list[tuple]
    allocatedPRB: list
    slotNumInFrame: int
    N_DMRS_ID: int
    lambda_bar: int
    n_SCID: int

class PDSCH_DMRS_Generator():
    def __init__(self, config:PDSCH_DMRS_GeneratorConfig) -> None:
        self.nOFDMSymbolsPerSlot = config.nOFDMSymbolsPerSlot
        self.slotNumInFrame = config.slotNumInFrame
        self.N_DMRS_ID = config.N_DMRS_ID
        self.lambda_bar = config.lambda_bar
        self.n_SCID = config.n_SCID
        self.SequenceGenerator = SequenceGenerator()
        self.DMRS_OFDMposition = self.__compute_DMRS_num_pos__(len(config.allocatedPRB), config.allocatedDMRSPerPRB)

    def __compute_DMRS_num_pos__(self, numAllocatedPRB: int, allocatedDMRSPerPRB: list[tuple]) -> list[tuple]:
        l_positions = {}
        for (k,l) in allocatedDMRSPerPRB:
            if l not in l_positions.keys():
                l_positions[l] = 1
            else:
                l_positions[l] += 1
        DMRS_OFDMposition = [(l, l_positions[l]*numAllocatedPRB) for l in l_positions]
        return DMRS_OFDMposition

    def __compute_c_init__(self, DMRS_OFDMposition: int):
        c_init = (
            (2**17)
            * (self.nOFDMSymbolsPerSlot * self.slotNumInFrame + DMRS_OFDMposition + 1)
            * (2 * self.N_DMRS_ID + 1) 
            + ((2**17) * (self.lambda_bar // 2))
            + (2 * self.N_DMRS_ID)
            + self.n_SCID
        ) % (2 ** 31)
        return c_init

    def process(self) -> dict:
        # DMRS_OFDMposition = [(l_0, numDMRS in l_0), (l_1, numDMRS in l_1), ...]
        DMRSs = {}
        for (l_position, numDMRS) in self.DMRS_OFDMposition:
            c_init = self.__compute_c_init__(l_position)
            c = self.SequenceGenerator.process(2*numDMRS, c_init)
            c = c.astype(np.int8)
            r = ((1 - 2*c[::2]) + 1j*(1 - 2*c[1::2])) / np.sqrt(2)
            DMRSs[l_position] = r
        return DMRSs
    
    
# def selectDMRSGenerator(transmission_type: TRANSMISSION_TYPE, config: PDSCH_DMRS_GeneratorConfig):
#     if transmission_type == TRANSMISSION_TYPE.PDSCH: return PDSCH_DMRS_Generator(config)
#     else: return PUSCH_DMRS_Generator(config)

if __name__ == "__main__":
    config = PDSCH_DMRS_GeneratorConfig(
        nOFDMSymbolsPerSlot = 14,
        allocatedDMRSPerPRB = [(0,2), (2,2), (4,2), (6,2), (8,2), (10,2)],
        allocatedPRB = [a for a in range(5,15)],
        slotNumInFrame = 0,
        N_DMRS_ID = 100,
        lambda_bar = 0,
        n_SCID = 0
    )

    Generator = PDSCH_DMRS_Generator(config)
    DMRSs = Generator.process()