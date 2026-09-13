from Transmitter import *
from Receiver import *
from Channel import *
from util import *
from EvaluationMetrics import *

class Simulator():
    def __init__(self, TxConfig: TransmitterConfig, RxConfig: ReceiverConfig, ChannelConfig: RayleighFadingConfig):
        self.Transmitter = Transmitter(TxConfig)
        self.ChannelConfig = ChannelConfig
        self.Receiver = Receiver(RxConfig)
        self.Evaluator = ChannelEstimationNMSE(TxConfig.allocatedPDSCHSymbols, TxConfig.allocatedPRB)
        self.ALLEvaluator = Evaluator(TxConfig.allocatedPDSCHSymbols,TxConfig.allocatedPRB)

    def process(self, DEBUG_MODE: bool, EsN0_dB: float, HARQ_number: int) -> dict:
        logging_level(DEBUG_MODE)
        rng = np.random.default_rng()
        RV_id = [0,2,3,1]
        InformationData = rng.integers(0, 2, size=100000, dtype=np.uint8)
        
        NDI = not self.Transmitter.NDI
        for rv_id in RV_id:
            TransmittedSymbols = self.Transmitter.process(InformationData, HARQ_number, NDI, rv_id)
            logging.debug("...................................")
            logging.debug("...... Transmitted Wave Form ......")
            self.Channel = RayleighFadingChannel(self.ChannelConfig)
            self.NoiseMixer = NoiseMixer(self.Receiver.NFFT, EsN0_dB)
            ChannelOutputs = self.Channel.process(TransmittedSymbols)
            ReceivedSignal, N0 = self.NoiseMixer.process(ChannelOutputs)
            logging.debug("...................................")
            HARQ_ACK, EstimatedTransportBlock = self.Receiver.process(ReceivedSignal, N0, HARQ_number, NDI, rv_id)
            if HARQ_ACK == "ACK":
                break
        
        Estimated = {
            "Channel": self.Receiver.meta["EstimatedChannel"],
            "QAMSymbols": self.Receiver.meta["EstimatedQAMSymbols"],
            "LLRs": self.Receiver.meta["LLRs"]
        }

        GroundTruth = {
            "Channel": self.Channel.ChannelFrequency,
            "QAMSymbols": self.Transmitter.meta["QAMSymbols"],
            "ScrambledBits": self.Transmitter.meta["ScrambledBits"]
        }

        results = self.ALLEvaluator.process(Estimated, GroundTruth)
        
        isSuccess = False
        if HARQ_ACK == "ACK":
            TransportBlock = InformationData[:self.Transmitter.meta["TBS"]]
            bit_errors = np.count_nonzero(TransportBlock != EstimatedTransportBlock)
            isSuccess = bit_errors == 0

        results["isSuccess"] =  isSuccess
        
        return results

if __name__ == '__main__':
    
    TxConfig = TransmitterConfig(
        nPRB = 50,
        allocatedPRB = [a for a in range(5,15)],
        allocatedPDSCHSymbols = [a for a in range(2,14)],
        allocatedDMRSPerPRB = [(0,2), (2,2), (4,2), (6,2), (8,2), (10,2)],
        nOFDMSymbolsPerSlot = 14,
        SubCarrierSpacing = int(30e3),
        Qm = 4,
        R = 0.3,
        nLayer = 1,
        nCodeWord = 1,
        rv_id = 0,
        nRNTI = 99,
        nID = 42,
        slotNumInFrame = 0,
        N_DMRS_ID = 100,
        lambda_bar = 0,
        n_SCID = 0,
        NFFT = 1024
    )

    RxConfig = ReceiverConfig(
        nPRB = 50,
        allocatedPRB = [a for a in range(5,15)],
        allocatedPDSCHSymbols = [a for a in range(2,14)],
        allocatedDMRSPerPRB = [(0,2), (2,2), (4,2), (6,2), (8,2), (10,2)],
        nOFDMSymbolsPerSlot = 14,
        SubCarrierSpacing = int(30e3),
        Qm = 4,
        R = 0.3,
        nLayer = 1,
        nCodeWord = 1,
        rv_id = 0,
        nRNTI = 99,
        nID = 42,
        slotNumInFrame = 0,
        N_DMRS_ID = 100,
        lambda_bar = 0,
        n_SCID = 0,
        NFFT = 1024,
        maxIter = 50
    )

    ChannelConfig = RayleighFadingConfig(
        velocity = 15,
        carrierFrequency = 2.5e9,
        delays_ns = [0, 100, 300],
        delayPower_dB = [0, -3, -6],
        Ts = 1 / (1024 * 30e3),         # 1 / Fs where Fs = NFFT * delta_f
        T_slot = 1 / 30e3,              # T_slot = 1 / delta_f
        nOFDMSymbolsPerSlot = 14,
        NFFT = 1024
    )
    
    DEBUG_MODE = False
    EsN0_dB = 0
    HARQ_number = 0

    ThisSimulator = Simulator(TxConfig, RxConfig, ChannelConfig)
    results = ThisSimulator.process(DEBUG_MODE, EsN0_dB, HARQ_number)
    if results["isSuccess"]:
        logging.info(f"===== Transmission: Success =====")
    else:
        logging.info(f"===== Transmission: Failure =====")
