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
    n_SCID: int
    NFFT: int

class Transmitter():
    def __init__(self, config: TransmitterConfig):
        logging.debug("..... Initializing configuration setup .....")
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
            NFFT = config.NFFT
        )
        self.OFDMModulator = OFDMModulator(temp_config)
        self.NDI = False
        self.HARQ_EncodedCodeBlocks = {}

    def process(self, InformationData, HARQ_number:int, NDI:bool, RV_id: int) -> list[np.ndarray]:
        if self.NDI != NDI:
            TransportBlock = InformationData[:self.meta["TBS"]]
            logging.debug("======== Completed generating transport block ========")
            CodeBlocks = self.CodeBlockSegmenter.process(TransportBlock, self.meta["LDPCBlockParam"])
            logging.debug("======== Completed code block segmentation ========")
            self.HARQ_EncodedCodeBlocks[HARQ_number] = self.Encoder.process(CodeBlocks, validityCheckFlag=False)
            logging.debug("======== Completed encoding code blocks ========")
            self.NDI = NDI
        
        RateMatchedCodeBlocks = self.RateMatcher.process(self.HARQ_EncodedCodeBlocks[HARQ_number], RV_id)
        logging.debug("======== Completed rate matching code blocks ========")
        Codeword = self.CodeBlockConcatenator.process(RateMatchedCodeBlocks)
        ScrambledBits = self.Scrambler.process(Codeword)
        self.meta["ScrambledBits"] = ScrambledBits
        logging.debug("======== Completed scrambling ========")
        QAMSymbols = self.QAMMapper.process(ScrambledBits)
        self.meta["QAMSymbols"] = QAMSymbols
        logging.debug("======== Completed QAM mapping ========")
        LayerMappedSymbols = self.LayerMapper.process(QAMSymbols)
        logging.debug("======== Completed layer mapping ========")
        DMRSs = self.DMRSGenerator.process()
        grid = self.ResourceMapper.process(LayerMappedSymbols, DMRSs)
        logging.debug("======== Completed constructing resource grid ========")
        TransmittedSymbols = self.OFDMModulator.process(grid)
        return TransmittedSymbols

    