from util import *
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
from Configuration import *

class Receiver():
    def __init__(self, config: NRSystemConfig):
        logging.debug("..... Initializing configuration setup .....")
        self.TBSGenerator = TBSGenerator()
        temp_config = TBSGeneratorConfig(
            numAllocatedPRB = len(config.bwp.allocatedPRB),
            numPDSCHSymbolsPerPRB = len(config.pdsch.allocatedSymbols),
            numDMRSPerPRB = len(config.dmrs.allocatedDMRSPerPRB),
            Qm = config.pdsch.Qm,
            R = config.pdsch.R,
            nLayer = config.pdsch.nLayer
        )
        self.meta = {}
        param = self.TBSGenerator.generate(temp_config)
        self.meta["TBS"] = param["TBS"]
        self.meta["G"] = param["G"]
        self.meta["baseGraph"] = selectLDPCBaseGraph(self.meta["TBS"], config.pdsch.R)
        self.CodeBlockSegmenter = CodeBlockSegmenter(self.meta["baseGraph"])
        self.meta["LDPCBlockParam"] = self.CodeBlockSegmenter.getLDPCBlockParam(self.meta["TBS"])

        dummyTransportBlock = np.array([0] * self.meta["TBS"])
        CodeBlocks = self.CodeBlockSegmenter.process(dummyTransportBlock, self.meta["LDPCBlockParam"])
        self.meta["mask_NULLs"] = [(CodeBlocks[i] == -1) for i in range(len(CodeBlocks))]

        temp_config = OFDMModulationConfig(
            nPRB = config.carrier.nPRB,
            nOFDMSymbolsPerSlot = config.carrier.nOFDMSymbolsPerSlot,
            SubcarrierSpacing = config.carrier.subcarrierSpacing,
            NFFT = config.carrier.NFFT
        )
        self.OFDMDemodulator = OFDMDemodulator(temp_config)
        temp_config = ResourceMappingConfig(
            nPRB = config.carrier.nPRB,
            allocatedPRB = config.bwp.allocatedPRB,
            allocatedPDSCHSymbols = config.pdsch.allocatedSymbols,
            allocatedDMRSPerPRB = config.dmrs.allocatedDMRSPerPRB,
            nLayer = config.pdsch.nLayer,
            nOFDMSymbolsPerSlot = config.carrier.nOFDMSymbolsPerSlot,
            slotNumInFrame = config.pdsch.slotNumInFrame,
            N_DMRS_ID = config.dmrs.N_DMRS_ID,
            lambda_bar = config.dmrs.lambda_bar,
            n_SCID = config.dmrs.n_SCID
        )
        self.ResourceDemapper = ResourceDemapper(temp_config)
        temp_config = PDSCH_DMRS_GeneratorConfig(
            nOFDMSymbolsPerSlot = config.carrier.nOFDMSymbolsPerSlot,
            allocatedDMRSPerPRB = config.dmrs.allocatedDMRSPerPRB,
            allocatedPRB = config.bwp.allocatedPRB,
            slotNumInFrame = config.pdsch.slotNumInFrame,
            N_DMRS_ID = config.dmrs.N_DMRS_ID,
            lambda_bar = config.dmrs.lambda_bar,
            n_SCID = config.dmrs.n_SCID
        )
        self.DMRSGenerator = PDSCH_DMRS_Generator(temp_config)
        temp_config = ChannelEstimatorConfig(
            allocatedPRB = config.bwp.allocatedPRB,
            allocatedPDSCHSymbols = config.pdsch.allocatedSymbols,
            allocatedDMRSPerPRB = config.dmrs.allocatedDMRSPerPRB,
            RETypeGrid = self.ResourceDemapper.RETypeGrid
        )

        self.ChannelEstimator = select_estimator(config.receiver.channelEstimatorType, temp_config)
        temp_config = EqualizerConfig(
            allocatedPRB=config.bwp.allocatedPRB,
            allocatedPDSCHSymbols=config.pdsch.allocatedSymbols
        )
        self.Equalizer = select_equalizer(config.receiver.equalizerType, temp_config)
        self.LayerDemapper = LayerDemapper(config.pdsch.nLayer)
        self.QAMDemapper = QAMDemapper(config.pdsch.Qm)
        self.Descrambler = PDSCHDescrambler(config.scrambling.nRNTI, config.scrambling.nID)
        temp_config = RateMatchingConfig(
            nLayer = config.pdsch.nLayer,
            Qm = config.pdsch.Qm,
            baseGraph = self.meta["baseGraph"],
            Z_c = self.meta["LDPCBlockParam"]["Z_c"],
            G = self.meta["G"]
        )
        self.rv_id = config.harq.rv_id
        self.RateRecoverer = RateRecoverer(temp_config, self.meta["LDPCBlockParam"]["C"], self.meta["mask_NULLs"])
        temp_config = LDPCConfig(
            baseGraph = self.meta["baseGraph"],
            i_LS = self.meta["LDPCBlockParam"]["i_LS"],
            Z_c = self.meta["LDPCBlockParam"]["Z_c"],
            K = self.meta["LDPCBlockParam"]["K"]
        )
        self.LDPCDecoder = LDPCDecoder(temp_config, config.receiver.maxLDPCIterations, self.meta["mask_NULLs"])
        self.CodeBlockCombiner = CodeBlockCombiner(self.meta["LDPCBlockParam"]["C"])
        self.NFFT = config.carrier.NFFT
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
        else:
            logging.debug("======== Failed decoding code blocks ========")
        HARQ_ACK, EstimatedTransportBlock = self.CodeBlockCombiner.process(EstimatedCodeBlocks, HARQ_ACK)
        if HARQ_ACK == "ACK":
            logging.debug("======== Completed estimating transport block ========")
        
        return HARQ_ACK, EstimatedTransportBlock