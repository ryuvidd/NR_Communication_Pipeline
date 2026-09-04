import math
from TBS_generator import *
from CRC import *
import numpy as np

LIFTING_SIZES = {
    "0": [2, 4, 8, 16, 32, 64, 128, 256],
    "1": [3, 6, 12, 24, 48, 96, 192, 384],
    "2": [5, 10, 20, 40, 80, 160, 320],
    "3": [7, 14, 28, 56, 112, 224],
    "4": [9, 18, 36, 72, 144, 288],
    "5": [11, 22, 44, 88, 176, 352],
    "6": [13, 26, 52, 104, 208],
    "7": [15, 30, 60, 120, 240]
}

def selectLDPCBaseGraph(A: int, R: float) -> int:
    if A <= 292: return 2
    elif R <= 0.25: return 2
    elif A <= 3824 and R <= 0.67: return 2
    else: return 1
    
class CodeBlockSegmenter():
    def __init__(self, baseGraph) -> None:
        if baseGraph in {1,2}: self.baseGraph = baseGraph
        else: raise ValueError("baseGraph must be 1 or 2")
        self.CRC24A = CRC(L="24A")
        self.CRC24B = CRC(L="24B")
    
    def __obtain_K_b__(self, B):
        if self.baseGraph == 1: return 22
        else:
            if B > 640: return 10
            elif B > 560: return 9
            elif B > 190: return 8
            else: return 6
    
    def __obtain_Z_c__(self, K_b, K_prime):
        Z_c = -1
        i_LS = -1
        for i_ls, Z_list in LIFTING_SIZES.items():
            for Z in Z_list:
                if K_b * Z >= K_prime:
                    if Z_c == -1 or Z < Z_c:
                        Z_c = Z
                        i_LS = i_ls
        return Z_c, i_LS
    
    def getLDPCBlockParam(self, TBS: int):
        ## Following TS38.212 clause 5.2.2: LDPC Code block segmentation ##
        B = TBS + 24
        K_cb = 8448 if self.baseGraph == 1 else 3840
        if B <= K_cb:
            C = 1
            L = 0
            B_prime = B
        else:
            L = 24
            C = math.ceil(B / (K_cb - L))
            B_prime = B + C * L

        K_prime = B_prime // C
        K_b = self.__obtain_K_b__(B)
        Z_c, i_LS = self.__obtain_Z_c__(K_b, K_prime)
        K = 22 * Z_c if self.baseGraph == 1 else 10 * Z_c
        F = K - K_prime
        LDPCBlockParam = {
            "K_cb": K_cb,
            "L": L,
            "C": C,
            "B_prime": B_prime,
            "K_prime": K_prime,
            "K_b": K_b,
            "Z_c": Z_c,
            "K": K,
            "F": F,
            "i_LS": i_LS
        }    
        return LDPCBlockParam
    
    def process(self, TransportBlock: np.ndarray, param: dict) -> list:
        TBwithCRC, crc = self.CRC24A.attachCRC(TransportBlock)
        B = len(TBwithCRC)

        codeBlocks = []
        C = param["C"]
        K = param["K"]
        K_prime = param["K_prime"]
        L = param["L"]
        LengthPayLoad = K_prime - L
        if LengthPayLoad != B // C:
            raise ValueError("Payload length is not equal to B / C.")
        for r in range(C):
            codeBlock = np.full(K, -1, dtype=np.int8)
            payLoad = TBwithCRC[r*LengthPayLoad:(r+1)*LengthPayLoad]
            if C > 1:
                CBwithCRC, crc = self.CRC24B.attachCRC(payLoad)
                codeBlock[:LengthPayLoad+24] = CBwithCRC
            else:
                codeBlock[:LengthPayLoad] = payLoad
            codeBlocks.append(codeBlock)
        return codeBlocks
    
class CodeBlockCombiner():
    def __init__(self, C):
        if C > 1:
            self.CRC24B = CRC(L="24B")
        self.CRC24A = CRC(L="24A")

    def process(self, EstimatedCodewords: list[np.ndarray]) -> tuple:
        retransmissionCodeBlockIndices = []
        EstimatedTransportBlock = np.array(-1)
        if len(EstimatedCodewords) > 1:
            ConcatentedCodeBlock = []
            for r,code_block in enumerate(EstimatedCodewords):
                if self.CRC24B.check(code_block):
                    ConcatentedCodeBlock.append(code_block[:-self.CRC24B.CRCLength])
                else:
                    retransmissionCodeBlockIndices.append(r)
            if len(retransmissionCodeBlockIndices) == 0:
                LastCRCCheckCodeBlock = np.concatenate(ConcatentedCodeBlock)
            else:
                return retransmissionCodeBlockIndices, EstimatedTransportBlock
        
        LastCRCCheckCodeBlock = EstimatedCodewords[0]
        if self.CRC24A.check(LastCRCCheckCodeBlock):
            EstimatedTransportBlock = LastCRCCheckCodeBlock[:-self.CRC24A.CRCLength]
        else:
            retransmissionCodeBlockIndices.append(-1)
        return retransmissionCodeBlockIndices, EstimatedTransportBlock
        
    
if __name__ == "__main__":
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
    TBS = param["TBS"]
    baseGraph = selectLDPCBaseGraph(TBS, config.R)
    ThisCodeBlockSegmenter = CodeBlockSegmenter(baseGraph)
    LDPCBlockParam = ThisCodeBlockSegmenter.getLDPCBlockParam(TBS)

    TransportBlock = np.array([1] * TBS)
    CodeBlocks = ThisCodeBlockSegmenter.process(TransportBlock, LDPCBlockParam)
    mask_NULLs = [(CodeBlocks[i] == -1) for i in range(len(CodeBlocks))]
