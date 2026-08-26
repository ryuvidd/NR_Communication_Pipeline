from TBS_generator import *
from CRC import *
from dataclasses import dataclass
from CBS import *
from LDPCEncoder import *
from RateMatching import *
import logging
from CBconcat import *
from Scrambling import *
from QAMModulators import *
from LayerMapping import *
from DMRSGeneration import *
from ResourceMapping import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass
class MainConfig:
    nPRB: int
    allocatedPRB: list
    allocatedPDSCHSymbols: list
    allocatedDMRSPerPRB: list[tuple]
    nOFDMSymbolsPerSlot: int
    Qm: int
    R: float
    nLayer: int
    nCodeWord: int
    nRNTI: int
    nID: int
    ModulatorType: QAM_MODULATION
    slotNumInFrame: int
    N_DMRS_ID: int
    lambda_bar: int
    n_SCID:int

class Transmitter():
    def __init__(self, config: MainConfig):
        logging.info("..... Initializing configuration setup .....")
        self.config = config
        self.TBSGenerator = TBSGenerator()
        temp_config = TBSGeneratorConfig(
            numAllocatedPRB = len(config.allocatedPRB),
            numPDSCHSymbolsPerPRB = len(config.allocatedPDSCHSymbols),
            numDMRSPerPRB = len(config.allocatedDMRSPerPRB),
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
        self.Scrambler = PDSCHScrambler(config.nRNTI, config.nCodeWord, config.nID)
        self.QAMModulator = SelectModulator(config.ModulatorType)
        self.LayerMapper = LayerMapper(config.nLayer)
        temp_config = PDSCH_DMRS_GeneratorConfig(
            nOFDMSymbolsPerSlot = config.nOFDMSymbolsPerSlot,
            allocatedDMRSPerPRB = config.allocatedDMRSPerPRB,
            allocatedPRB = config.allocatedPRB,
            slotNumInFrame = config.slotNumInFrame,
            N_DMRS_ID = config.N_DMRS_ID,
            lambda_bar = config.lambda_bar,
            n_SCID = config.n_SCID
        )
        self.DMRSGenerator = PDSCH_DMRS_Generator(temp_config)
        temp_config = ResourceMappingConfig(
            nPRB = config.nPRB,
            allocatedPRB = config.allocatedPRB,
            allocatedPDSCHSymbols = config.allocatedPDSCHSymbols,
            allocatedDMRSPerPRB = config.allocatedDMRSPerPRB,
            nLayer = config.nLayer,
            nOFDMSymbolsPerSlot = config.nOFDMSymbolsPerSlot,
            slotNumInFrame = config.slotNumInFrame,
            N_DMRS_ID = config.N_DMRS_ID,
            lambda_bar = config.lambda_bar,
            n_SCID = config.n_SCID
        )
        self.ResourceMapper = ResourceMapper(temp_config)

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
        QAMSymbols = self.QAMModulator.modulate(ScrambledBits)
        logging.info("======== Completed QAM mapping ========")
        LayerMappedSymbols = self.LayerMapper.process(QAMSymbols)
        logging.info("======== Completed layer mapping ========")
        DMRSs = self.DMRSGenerator.process()
        grid = self.ResourceMapper.process(LayerMappedSymbols, DMRSs)
        logging.info("======== Completed constructing resource grid ========")
        return grid
    
if __name__ == '__main__':
    mainConfig = MainConfig(
        nPRB = 50,
        allocatedPRB = [a for a in range(5,15)],
        allocatedPDSCHSymbols = [a for a in range(2,14)],
        allocatedDMRSPerPRB = [(0,2), (2,2), (4,2), (6,2), (8,2), (10,2)],
        nOFDMSymbolsPerSlot = 14,
        Qm = 2,
        R = 0.5,
        nLayer = 1,
        nCodeWord = 1,
        nRNTI = 99,
        nID = 42,
        ModulatorType = QAM_MODULATION.QPSK_GRAY,
        slotNumInFrame = 0,
        N_DMRS_ID = 100,
        lambda_bar = 0,
        n_SCID = 0
    )

    ThisTransmitter = Transmitter(mainConfig)
    TransmittedWaveForm = ThisTransmitter.process()
    logging.info("===== Success =====")

    