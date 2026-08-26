from dataclasses import dataclass
import math

TBS_TABLE = [
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
    112, 120, 128, 136, 144, 152, 160, 168, 176, 
    184, 192, 208, 224, 240, 256, 272, 288, 304, 
    320, 336, 352, 368, 384, 408, 432, 456, 480, 
    504, 528, 552, 576, 608, 640, 672, 704, 736, 
    768, 808, 848, 888, 928, 984, 1032, 1064, 
    1128, 1160, 1192, 1224, 1256, 1288, 1320, 
    1352, 1416, 1480, 1544, 1608, 1672, 1736, 
    1800, 1864, 1928, 2024, 2088, 2152, 2216, 
    2280, 2408, 2472, 2536, 2600, 2664, 2728, 
    2792, 2856, 2976, 3104, 3240, 3368, 3496, 
    3624, 3752, 3824
]

@dataclass
class TBSGeneratorConfig:
    numAllocatedPRB: int
    numPDSCHSymbolsPerPRB: int
    numDMRSPerPRB: int
    Qm: int
    R: float
    nLayer: int

class TBSGenerator():
    def __TBS_table5_1_3_2_1__(self, N_info_prime: int):
        for i in TBS_TABLE:
            if N_info_prime < i: 
                return i
        raise ValueError("N_info_prime is greater than 3,824.")
    
    def generate(self, config: TBSGeneratorConfig):
        ## Following TS38.214 clause 5.1.3.2 (PDSCH) and 6.1.4.2 (PUSCH) ##
        N_RE = config.numAllocatedPRB * min(156, (config.numPDSCHSymbolsPerPRB * 12 - config.numDMRSPerPRB))
        G = N_RE * config.Qm * config.nLayer
        N_info = G * config.R
        if N_info <= 3824:
            n = max(3, math.floor(math.log2(N_info)) - 6)
            N_info_prime = max(24, 2**n * math.floor(N_info / 2**n))
            TBS = self.__TBS_table5_1_3_2_1__(N_info_prime)
        else:
            n = math.floor(math.log2(N_info - 24)) - 5
            N_info_prime = max(3840, 2**n * round((N_info - 24) / 2**n))
            if config.R <= 0.25:
                C = math.ceil((N_info_prime + 24) / 3816)
            else:
                if N_info_prime > 8424:
                    C = math.ceil((N_info_prime + 24) / 8424)
                else: 
                    C = 1
            TBS = 8 * C * math.ceil((N_info_prime + 24) / (8 * C)) - 24
        return {"TBS": TBS, 
                "G": G
        }
    
if __name__ == '__main__':
    config = TBSGeneratorConfig(
        numAllocatedPRB = 50,
        numPDSCHSymbolsPerPRB = 12,
        numDMRSPerPRB = 12,
        Qm = 4,
        R = 0.5,
        nLayer = 1
    )
    ThisTBSGenerator = TBSGenerator()
    param = ThisTBSGenerator.generate(config)
    print('TBS: ', param["TBS"])
    print('C: ', param["C"])
    print('G: ', param["G"])