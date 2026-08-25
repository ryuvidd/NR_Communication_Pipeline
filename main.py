from TBS_generator import *
from CRC import *
from dataclasses import dataclass
from CBS import *
from LDPCEncoder import *
from RateMatching import *
import logging
from CBconcat import *
from Scrambling import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class MainConfig:
    nPRB: int
    nSymbolsPerPRB: int
    nDMRSPerPRB: int
    Qm: int
    R: float
    nLayer: int
    n_RNTI: int
    n_ID: int

class Transmitter():
    def __init__(self, config: MainConfig):
        logging.info("..... Initializing configuration setup .....")
        self.config = config
        self.TBSGenerator = TBSGenerator()
        temp_config = TBSGeneratorConfig(
            nPRB = config.nPRB,
            nSymbolsPerPRB = config.nSymbolsPerPRB,
            nDMRSPerPRB = config.nDMRSPerPRB,
            Qm = config.Qm,
            R = config.R,
            nLayer = config.nLayer
        )
        self.meta = {}
        param = self.TBSGenerator.generate(temp_config)
        self.meta["TBS"] = param["TBS"]
        self.meta["G"] = param["G"]
        self.meta["baseGraph"] = selectLDPCBaseGraph(self.meta["TBS"], config.R)
        self.CodeBlockSegmenter = CodeBlockSegmenter(self.meta["baseGraph"])
        self.meta["LDPCBlockParam"] = self.CodeBlockSegmenter.getLDPCBlockParam(self.meta["TBS"])
        temp_config = LDPCConfig(
            baseGraph = self.meta["baseGraph"],
            i_LS = self.meta["LDPCBlockParam"]["i_LS"],
            Z_c = self.meta["LDPCBlockParam"]["Z_c"],
            K = self.meta["LDPCBlockParam"]["K"],
            K_b = self.meta["LDPCBlockParam"]["K_b"]
        )
        self.Encoder = LDPCEncoder(temp_config)
        temp_config = RateMatchConfig(
            nLayer = config.nLayer,
            Qm = config.Qm,
            baseGraph = self.meta["baseGraph"],
            Z_c = self.meta["LDPCBlockParam"]["Z_c"],
            G = self.meta["G"]
        )
        self.RateMatcher = RateMatcher(temp_config)
        self.CodeBlockConcatenator = CodeBlockConcatenator()
        self.Scrambler = PDSCHScrambler(n_RNTI=config.n_RNTI, q=0, n_ID=config.n_ID)


    def process(self) -> np.ndarray:
        rng = np.random.default_rng()
        TransportBlock = rng.integers(0, 2, size=self.meta["TBS"], dtype=np.int8)
        logging.info("======== Completed generating transport block ========")
        CodeBlocks = self.CodeBlockSegmenter.generateCodeBlocks(TransportBlock, self.meta["LDPCBlockParam"])
        logging.info("======== Completed code block segmentation ========")
        EncodedCodeBlocks = self.Encoder.encode(CodeBlocks, validityCheckFlag=False)
        logging.info("======== Completed encoding code blocks ========")
        RateMatchedCodeBlocks = self.RateMatcher.process(EncodedCodeBlocks, rv_id=1)
        logging.info("======== Completed rate matching code blocks ========")
        Codeword = self.CodeBlockConcatenator.process(RateMatchedCodeBlocks)
        ScrambledBits = self.Scrambler.process(Codeword)
        logging.info("======== Completed scrambling ========")
        return ScrambledBits
    
if __name__ == '__main__':
    mainConfig = MainConfig(
        nPRB = 50,
        nSymbolsPerPRB = 12,
        nDMRSPerPRB =12,
        Qm = 4,
        R = 0.5,
        nLayer = 1,
        n_RNTI = 99,
        n_ID = 42,
    )

    ThisTransmitter = Transmitter(mainConfig)
    Codeword = ThisTransmitter.process()
    logging.info("===== Success =====")

    