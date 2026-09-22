from Transmitter import *
from Receiver import *
from Channel import *
from util import *
from EvaluationMetrics import *
from Configuration import *

class Simulator():
    def __init__(self, configs: list[NRSystemConfig]):
        config = configs[0]
        self.Transmitter = Transmitter(config)
        self.Channel = select_channel_model(config.channel)
        self.Receiver01 = Receiver(configs[0])
        self.Receiver02 = Receiver(configs[1])
        self.Receiver03 = Receiver(configs[2])
        self.Receiver04 = Receiver(configs[3])
        self.Evaluators = Evaluator(config.pdsch.allocatedSymbols, config.bwp.allocatedPRB)

    def reset(self):
        self.Transmitter.HARQ_EncodedCodeBlocks = {}
        self.Receiver01.RateRecoverer.HARQ_buffer = {}
        self.Receiver02.RateRecoverer.HARQ_buffer = {}
        self.Receiver03.RateRecoverer.HARQ_buffer = {}
        self.Receiver04.RateRecoverer.HARQ_buffer = {}
        NDI = not self.Transmitter.NDI
        return NDI

    def process(self, DEBUG_MODE: bool, EsN0_dB: float, HARQ_number: int) -> list[dict]:
        logging_level(DEBUG_MODE)
        RV_id = [0]
        InformationData = np.random.randint(0, 2, size=100000, dtype=np.uint8)
        NDI = self.reset()
        HARQ_ACKs = ["NACK"] * 4

        for rv_id in RV_id:
            meta = {}
            TransmittedSymbols = self.Transmitter.process(InformationData, HARQ_number, NDI, rv_id)
            logging.debug("...................................")
            logging.debug("...... Transmitted Wave Form ......")
            channel_input = {
                "TransmittedSymbols": TransmittedSymbols,
                "Ts": 1/self.Transmitter.OFDMModulator.Fs,
                "subcarrierSpacing": self.Transmitter.OFDMModulator.delta_f,
                "NFFT": self.Transmitter.OFDMModulator.NFFT
            }
            ChannelOutputs = self.Channel.process(channel_input)
            self.AWGNChannel = AWGNChannel(self.Transmitter.OFDMModulator.NFFT, EsN0_dB)
            ReceivedSignal, NoiseVar = self.AWGNChannel.process(ChannelOutputs)
            meta["mean_h"] = self.Channel.mean_h
            meta["R_hh"] = self.Channel.R_hh
            meta["NoiseVar"] = NoiseVar
            logging.debug("...................................")
            if HARQ_ACKs[0] == "NACK":
                HARQ_ACKs[0], EstimatedTransportBlock01 = self.Receiver01.process(ReceivedSignal, HARQ_number, NDI, rv_id, meta)
            if HARQ_ACKs[1] == "NACK":
                HARQ_ACKs[1], EstimatedTransportBlock02 = self.Receiver02.process(ReceivedSignal, HARQ_number, NDI, rv_id, meta)
            if HARQ_ACKs[2] == "NACK":
                HARQ_ACKs[2], EstimatedTransportBlock03 = self.Receiver03.process(ReceivedSignal, HARQ_number, NDI, rv_id, meta)
            if HARQ_ACKs[3] == "NACK":
                HARQ_ACKs[3], EstimatedTransportBlock04 = self.Receiver04.process(ReceivedSignal, HARQ_number, NDI, rv_id, meta)
            
            if all(status == "ACK" for status in HARQ_ACKs):
                break
        
        Estimated01 = {
            "Channel": self.Receiver01.meta["EstimatedChannel"],
            "QAMSymbols": self.Receiver01.meta["EstimatedQAMSymbols"],
            "LLRs": self.Receiver01.meta["LLRs"],
            "TransportBlock": EstimatedTransportBlock01
        }
        Estimated02 = {
            "Channel": self.Receiver02.meta["EstimatedChannel"],
            "QAMSymbols": self.Receiver02.meta["EstimatedQAMSymbols"],
            "LLRs": self.Receiver02.meta["LLRs"],
            "TransportBlock": EstimatedTransportBlock02
        }
        Estimated03 = {
            "Channel": self.Receiver03.meta["EstimatedChannel"],
            "QAMSymbols": self.Receiver03.meta["EstimatedQAMSymbols"],
            "LLRs": self.Receiver03.meta["LLRs"],
            "TransportBlock": EstimatedTransportBlock03
        }
        Estimated04 = {
            "Channel": self.Receiver04.meta["EstimatedChannel"],
            "QAMSymbols": self.Receiver04.meta["EstimatedQAMSymbols"],
            "LLRs": self.Receiver04.meta["LLRs"],
            "TransportBlock": EstimatedTransportBlock04
        }

        GroundTruth = {
            "Channel": self.Channel.ChannelFrequency,
            "QAMSymbols": self.Transmitter.meta["QAMSymbols"],
            "ScrambledBits": self.Transmitter.meta["ScrambledBits"],
            "TransportBlock": InformationData[:self.Transmitter.meta["TBS"]]
        }

        results01 = self.Evaluators.process(Estimated01, GroundTruth)
        results02 = self.Evaluators.process(Estimated02, GroundTruth)
        results03 = self.Evaluators.process(Estimated03, GroundTruth)
        results04 = self.Evaluators.process(Estimated04, GroundTruth)
        results = [results01, results02, results03, results04]

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
        receiver=receiver01Config,
    )

    DEBUG_MODE = False
    EsN0_dB = 0
    HARQ_number = 0

    ThisSimulator = Simulator([config])
    results = ThisSimulator.process(DEBUG_MODE, EsN0_dB, HARQ_number)
    if results[0]["bler"] == 0:
        logging.info(f"===== Transmission: Success =====")
    else:
        logging.info(f"===== Transmission: Failure =====")