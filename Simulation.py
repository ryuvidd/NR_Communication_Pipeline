from Transmitter import *
from Receiver import *
from Channel import *
from util import *
from EvaluationMetrics import *
from Configuration import *

class Simulator():
    def __init__(self, configs: list[NRSystemConfig], compare_RVs_flag: bool):
        config = configs[0]
        self.Transmitter = Transmitter(config)
        self.Channel = select_channel_model(config.channel)
        self.Receivers = []
        if compare_RVs_flag:
            self.numVariations = 4
            self.Receivers.append(Receiver(configs[0]))
        else:
            self.numVariations = len(configs)
            for i in range(self.numVariations):
                self.Receivers.append(Receiver(configs[i]))
        self.Evaluators = Evaluator(config.pdsch.allocatedSymbols, config.bwp.allocatedPRB, compare_RVs_flag)
        self.RV_id = config.harq.rv_id
        self.compare_RVs_flag = compare_RVs_flag

    def reset(self):
        self.Transmitter.HARQ_EncodedCodeBlocks = {}
        for i in range(len(self.Receivers)):
            self.Receivers[i].RateRecoverer.HARQ_buffer = {}
        NDI = not self.Transmitter.NDI
        return NDI
    
    def evaluate_results(self, EstimatedTransportBlocks, InformationData):
        Estimateds = []
        for i in range(self.numVariations):
            Estimated_results = {
                "Channel": self.Receivers[i].meta["EstimatedChannel"],
                "QAMSymbols": self.Receivers[i].meta["EstimatedQAMSymbols"],
                "LLRs": self.Receivers[i].meta["LLRs"],
                "TransportBlock": EstimatedTransportBlocks[i]
            }
            Estimateds.append(Estimated_results)

        GroundTruth = {
            "Channel": self.Channel.ChannelFrequency,
            "QAMSymbols": self.Transmitter.meta["QAMSymbols"],
            "ScrambledBits": self.Transmitter.meta["ScrambledBits"],
            "TransportBlock": InformationData[:self.Transmitter.meta["TBS"]]
        }

        results = []
        for i in range(self.numVariations):
            results.append(self.Evaluators.process(Estimateds[i], GroundTruth))
        
        return results
    
    def evaluate_results_RVs(self, EstimatedTransportBlocks, InformationData):
        Estimateds = []
        for i in range(self.numVariations):
            Estimated_results = {
                "TransportBlock": EstimatedTransportBlocks[i]
            }
            Estimateds.append(Estimated_results)

        GroundTruth = {
            "TransportBlock": InformationData[:self.Transmitter.meta["TBS"]]
        }

        results = []
        for i in range(self.numVariations):
            results.append(self.Evaluators.process(Estimateds[i], GroundTruth))
        
        return results

    def process(self, DEBUG_MODE: bool, EsN0_dB: float, HARQ_number: int) -> list[dict]:
        logging_level(DEBUG_MODE)
        InformationData = np.random.randint(0, 2, size=100000, dtype=np.uint8)
        NDI = self.reset()
        HARQ_ACKs = ["NACK"] * self.numVariations
        EstimatedTransportBlocks = [0] * self.numVariations
        results = []

        for rv_idx, rv_id in enumerate(self.RV_id):
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
            if self.compare_RVs_flag:
                HARQ_ACKs[rv_idx], EstimatedTransportBlocks[rv_idx] = self.Receivers[0].process(ReceivedSignal, HARQ_number, NDI, rv_id, meta)
            else:
                for i in range(self.numVariations):
                    if HARQ_ACKs[i] == "NACK":
                        HARQ_ACKs[i], EstimatedTransportBlocks[i] = self.Receivers[i].process(ReceivedSignal, HARQ_number, NDI, rv_id, meta)
                
                if all(status == "ACK" for status in HARQ_ACKs):
                    break
        
        if self.compare_RVs_flag:
            results = self.evaluate_results_RVs(EstimatedTransportBlocks, InformationData)
        else:
            results = self.evaluate_results(EstimatedTransportBlocks, InformationData)

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

    ThisSimulator = Simulator([config], True)
    results = ThisSimulator.process(DEBUG_MODE, EsN0_dB, HARQ_number)
    if results[0]["bler"] == 0:
        logging.info(f"===== Transmission: Success =====")
    else:
        logging.info(f"===== Transmission: Failure =====")