from util import *
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
from Configuration import *

class Transmitter():
    def __init__(self, config:NRSystemConfig):
        logging.debug("..... Initializing configuration setup .....")
        self.meta = {}
        self.TBSGenerator = TBSGenerator()
        temp_config = TBSGeneratorConfig(
            numAllocatedPRB = len(config.bwp.allocatedPRB),
            numPDSCHSymbolsPerPRB = len(config.pdsch.allocatedSymbols),
            numDMRSPerPRB = len(config.dmrs.allocatedDMRSPerPRB),
            Qm = config.pdsch.Qm,
            R = config.pdsch.R,
            nLayer = config.pdsch.nLayer
        )
        param = self.TBSGenerator.generate(temp_config)
        self.meta["TBS"] = param["TBS"]
        self.meta["G"] = param["G"]
        self.meta["baseGraph"] = selectLDPCBaseGraph(self.meta["TBS"], config.pdsch.R)
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
            nLayer = config.pdsch.nLayer,
            Qm = config.pdsch.Qm,
            baseGraph = self.meta["baseGraph"],
            Z_c = self.meta["LDPCBlockParam"]["Z_c"],
            G = self.meta["G"]
        )
        self.rv_id = config.harq.rv_id
        self.RateMatcher = RateMatcher(temp_config)
        self.CodeBlockConcatenator = CodeBlockConcatenator()
        self.Scrambler = PDSCHScrambler(config.scrambling.nRNTI, config.scrambling.nID)
        self.QAMMapper = QAMMapper(config.pdsch.Qm)
        self.LayerMapper = LayerMapper(config.pdsch.nLayer)
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
        self.ResourceMapper = ResourceMapper(temp_config)
        temp_config = OFDMModulationConfig(
            nPRB = config.carrier.nPRB,
            nOFDMSymbolsPerSlot = config.carrier.nOFDMSymbolsPerSlot,
            SubcarrierSpacing = config.carrier.subcarrierSpacing,
            NFFT = config.carrier.NFFT
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

    