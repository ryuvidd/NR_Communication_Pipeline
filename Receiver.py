from util import *
from dataclasses import dataclass
from OFDMModulation import *
from ResourceMapping import *
from ChannelEstimator import *
from Equalizer import *
from LayerMapping import *
from QAMMapping import *
from Scrambling import *
from TBS_generator import *
from LDPC_CBS import *
from LDPCEncoder import *
from LDPC_RateMatching_Recovery import *

@dataclass
class ReceiverConfig:
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
    maxIter: int

class Receiver():
    def __init__(self, config: ReceiverConfig):
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

        dummyTransportBlock = np.array([0] * self.meta["TBS"])
        CodeBlocks = self.CodeBlockSegmenter.process(dummyTransportBlock, self.meta["LDPCBlockParam"])
        self.meta["mask_NULLs"] = [(CodeBlocks[i] == -1) for i in range(len(CodeBlocks))]

        temp_config = OFDMModulationConfig(
            nPRB = config.nPRB,
            nOFDMSymbolsPerSlot = config.nOFDMSymbolsPerSlot,
            SubcarrierSpacing = config.SubCarrierSpacing,
            NFFT = config.NFFT
        )
        self.OFDMDemodulator = OFDMDemodulator(temp_config)
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
        self.ResourceDemapper = ResourceDemapper(temp_config)
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
        temp_config = ChannelEstimatorConfig(
            allocatedPRB = config.allocatedPRB,
            allocatedPDSCHSymbols = config.allocatedPDSCHSymbols,
            allocatedDMRSPerPRB = config.allocatedDMRSPerPRB,
            RETypeGrid = self.ResourceDemapper.RETypeGrid
        )
        self.ChannelEstimator = LSEstimator(temp_config)
        self.Equalizer = ZeroForcingEqualizer(config.allocatedPRB, config.allocatedPDSCHSymbols)
        self.LayerDemapper = LayerDemapper(config.nLayer)
        self.QAMDemapper = QAMDemapper(config.Qm)
        self.Descrambler = PDSCHDescrambler(config.nRNTI, config.nCodeWord, config.nID)
        temp_config = RateMatchingConfig(
            nLayer = config.nLayer,
            Qm = config.Qm,
            baseGraph = self.meta["baseGraph"],
            Z_c = self.meta["LDPCBlockParam"]["Z_c"],
            G = self.meta["G"]
        )
        self.rv_id = config.rv_id
        self.RateRecoverer = RateRecoverer(temp_config, self.meta["LDPCBlockParam"]["C"], self.meta["mask_NULLs"])
        temp_config = LDPCConfig(
            baseGraph = self.meta["baseGraph"],
            i_LS = self.meta["LDPCBlockParam"]["i_LS"],
            Z_c = self.meta["LDPCBlockParam"]["Z_c"],
            K = self.meta["LDPCBlockParam"]["K"]
        )
        self.LDPCDecoder = LDPCDecoder(temp_config, config.maxIter, self.meta["mask_NULLs"])
        self.CodeBlockCombiner = CodeBlockCombiner(self.meta["LDPCBlockParam"]["C"])
        self.NFFT = config.NFFT
        self.NDI = False

    def process(self, ReceivedSignal: list[np.ndarray], N0: float, HARQ_number, NDI, RV_id) -> tuple:
        EstimatedGrid = self.OFDMDemodulator.process(ReceivedSignal)
        logging.debug("======== Completed restructing resource grid ========")
        DMRSs = self.DMRSGenerator.process()
        EstimatedChannel = self.ChannelEstimator.process(EstimatedGrid, DMRSs)
        self.meta["EstimatedChannel"] = EstimatedChannel
        logging.debug("======== Completed estimating channel ========")
        EqualizedGrid, EffectiveN0 = self.Equalizer.process(EstimatedChannel, EstimatedGrid, N0)
        logging.debug("======== Completed equalizing ========")
        EstimatedLayerMappedSymbols, EffectiveN0 = self.ResourceDemapper.process(EqualizedGrid, EffectiveN0)
        logging.debug("======== Completed restrucing layered symbols ========")
        EstimatedQAMSymbols, EffectiveN0 = self.LayerDemapper.process(EstimatedLayerMappedSymbols, EffectiveN0)
        self.meta["EstimatedQAMSymbols"] = EstimatedQAMSymbols
        logging.debug("======== Completed estimating QAM symbols ========")
        LLRs = self.QAMDemapper.process(EstimatedQAMSymbols, EffectiveN0)
        self.meta["LLRs"] = LLRs
        logging.debug("======== Completed estimating LLRs ========")
        DescrambledLLRs = self.Descrambler.process(LLRs)
        logging.debug("======== Completed descrambling LLRs ========")
        if self.NDI != NDI:
            self.RateRecoverer.reset_buffer(HARQ_number)
            self.NDI = NDI
        RateRecoveredLLRs = self.RateRecoverer.process(DescrambledLLRs, HARQ_number, RV_id)
        logging.debug("======== Completed rate recovery LLRs ========")
        HARQ_ACK, EstimatedCodeBlocks = self.LDPCDecoder.process(RateRecoveredLLRs)
        if HARQ_ACK == "ACK":
            logging.debug("======== Completed decoding code blocks ========")
            HARQ_ACK, EstimatedTransportBlock = self.CodeBlockCombiner.process(EstimatedCodeBlocks)
            logging.debug("======== Completed estimating transport block ========")
            return HARQ_ACK, EstimatedTransportBlock
        else:
            logging.debug("======== Failed decoding code blocks ========")
            return HARQ_ACK, np.array(-1)
        
