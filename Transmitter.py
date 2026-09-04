from util import *
from dataclasses import dataclass
from TBS_generator import *
from CRC import *
from LDPC_CBS import *
from LDPCEncoder import *
from LDPC_RateMatching_Recovery import *
from CBconcat import *
from Scrambling import *
from QAMMapping import *
from LayerMapping import *
from DMRSGeneration import *
from ResourceMapping import *
from OFDMModulation import *

@dataclass
class TransmitterConfig:
    nPRB: int
    allocatedPRB: list
    allocatedPDSCHSymbols: list
    allocatedDMRSPerPRB: list[tuple]
    nOFDMSymbolsPerSlot: int
    SubCarrierSpacing: int
    Qm: int
    R: float
    nLayer: int
    nCodeWord: int
    rv_id: int
    nRNTI: int
    nID: int
    slotNumInFrame: int
    N_DMRS_ID: int
    lambda_bar: int
    n_SCID:int

class Transmitter():
    def __init__(self, config: TransmitterConfig):
        logging.info("..... Initializing configuration setup .....")
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
            K = self.meta["LDPCBlockParam"]["K"]
        )
        self.Encoder = LDPCEncoder(temp_config)
        temp_config = RateMatchingConfig(
            nLayer = config.nLayer,
            Qm = config.Qm,
            baseGraph = self.meta["baseGraph"],
            Z_c = self.meta["LDPCBlockParam"]["Z_c"],
            G = self.meta["G"]
        )
        self.rv_id = config.rv_id
        self.RateMatcher = RateMatcher(temp_config)
        self.CodeBlockConcatenator = CodeBlockConcatenator()
        self.Scrambler = PDSCHScrambler(config.nRNTI, config.nCodeWord, config.nID)
        self.QAMMapper = QAMMapper(config.Qm)
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
        temp_config = OFDMModulationConfig(
            nPRB = config.nPRB,
            nOFDMSymbolsPerSlot = config.nOFDMSymbolsPerSlot,
            SubcarrierSpacing = config.SubCarrierSpacing,
            NFFT = 1024
        )
        self.OFDMModulator = OFDMModulator(temp_config)

    def process(self, InformationData) -> np.ndarray:
        TransportBlock = InformationData[:self.meta["TBS"]]
        logging.info("======== Completed generating transport block ========")
        CodeBlocks = self.CodeBlockSegmenter.process(TransportBlock, self.meta["LDPCBlockParam"])
        logging.info("======== Completed code block segmentation ========")
        EncodedCodeBlocks = self.Encoder.process(CodeBlocks, validityCheckFlag=False)
        logging.info("======== Completed encoding code blocks ========")
        RateMatchedCodeBlocks = self.RateMatcher.process(EncodedCodeBlocks, self.rv_id)
        logging.info("======== Completed rate matching code blocks ========")
        Codeword = self.CodeBlockConcatenator.process(RateMatchedCodeBlocks)
        ScrambledBits = self.Scrambler.process(Codeword)
        logging.info("======== Completed scrambling ========")
        QAMSymbols = self.QAMMapper.process(ScrambledBits)
        logging.info("======== Completed QAM mapping ========")
        LayerMappedSymbols = self.LayerMapper.process(QAMSymbols)
        logging.info("======== Completed layer mapping ========")
        DMRSs = self.DMRSGenerator.process()
        grid = self.ResourceMapper.process(LayerMappedSymbols, DMRSs)
        logging.info("======== Completed constructing resource grid ========")
        TransmittedWaveForm = self.OFDMModulator.process(grid)
        return TransmittedWaveForm
    
if __name__ == '__main__':
    config = TransmitterConfig(
        nPRB = 50,
        allocatedPRB = [a for a in range(5,15)],
        allocatedPDSCHSymbols = [a for a in range(2,14)],
        allocatedDMRSPerPRB = [(0,2), (2,2), (4,2), (6,2), (8,2), (10,2)],
        nOFDMSymbolsPerSlot = 14,
        SubCarrierSpacing = int(30e3),
        Qm = 2,
        R = 0.5,
        nLayer = 1,
        nCodeWord = 1,
        rv_id = 1,
        nRNTI = 99,
        nID = 42,
        slotNumInFrame = 0,
        N_DMRS_ID = 100,
        lambda_bar = 0,
        n_SCID = 0
    )
    rng = np.random.default_rng(34)
    InformationData = rng.integers(0, 2, size=100000, dtype=np.uint8)

    ThisTransmitter = Transmitter(config)
    TransmittedWaveForm = ThisTransmitter.process(InformationData)
    logging.info("===== Success =====")

    