from Transmitter import *
from Receiver import *
from Channel import *
from util import *
from EvaluationMetrics import *
from Configuration import *

class Simulator():
    def __init__(self, config: NRSystemConfig):
        self.Transmitter = Transmitter(config)
        self.Channel = select_channel_model(config.channel)
        self.Receiver = Receiver(config)
        self.Evaluators = Evaluator(config.pdsch.allocatedSymbols, config.bwp.allocatedPRB)

    def reset(self):
        self.Transmitter.HARQ_EncodedCodeBlocks = {}
        self.Receiver.RateRecoverer.HARQ_buffer = {}
        NDI = not self.Transmitter.NDI
        return NDI

    def process(self, DEBUG_MODE: bool, EsN0_dB: float, HARQ_number: int) -> dict:
        logging_level(DEBUG_MODE)
        rng = np.random.default_rng()
        RV_id = [0,2,3,1]
        InformationData = rng.integers(0, 2, size=100000, dtype=np.uint8)
        NDI = self.reset()

        for rv_id in RV_id:
            TransmittedSymbols = self.Transmitter.process(InformationData, HARQ_number, NDI, rv_id)
            logging.debug("...................................")
            logging.debug("...... Transmitted Wave Form ......")
            ChannelOutputs = self.Channel.process(TransmittedSymbols, Ts=1/self.Transmitter.OFDMModulator.Fs, subcarrierSpacing=self.Transmitter.OFDMModulator.delta_f, NFFT=self.Transmitter.OFDMModulator.NFFT)
            self.AWGNChannel = AWGNChannel(self.Receiver.NFFT, EsN0_dB)
            ReceivedSignal, N0 = self.AWGNChannel.process(ChannelOutputs)
            logging.debug("...................................")
            HARQ_ACK, EstimatedTransportBlock = self.Receiver.process(ReceivedSignal, N0, HARQ_number, NDI, rv_id)
            if HARQ_ACK == "ACK":
                break
        
        Estimated = {
            "Channel": self.Receiver.meta["EstimatedChannel"],
            "QAMSymbols": self.Receiver.meta["EstimatedQAMSymbols"],
            "LLRs": self.Receiver.meta["LLRs"],
            "TransportBlock": EstimatedTransportBlock
        }

        GroundTruth = {
            "Channel": self.Channel.ChannelFrequency,
            "QAMSymbols": self.Transmitter.meta["QAMSymbols"],
            "ScrambledBits": self.Transmitter.meta["ScrambledBits"],
            "TransportBlock": InformationData[:self.Transmitter.meta["TBS"]]
        }

        results = self.Evaluators.process(Estimated, GroundTruth)
        
        return results

if __name__ == '__main__':

    config = NRSystemConfig(
        carrier=carrierConfig,
        bwp=bwpConfig,
        pdsch=pdschConfig,
        dmrs=dmrsConfig,
        scrambling=scramblingConfig,
        harq=harqConfig,
        channel=channelConfig,
        receiver=receiverConfig,
    )

    DEBUG_MODE = False
    EsN0_dB = 0
    HARQ_number = 0

    ThisSimulator = Simulator(config)
    results = ThisSimulator.process(DEBUG_MODE, EsN0_dB, HARQ_number)
    if results["bler"] == 0:
        logging.info(f"===== Transmission: Success =====")
    else:
        logging.info(f"===== Transmission: Failure =====")