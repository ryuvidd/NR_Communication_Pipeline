import numpy as np
import logging

class ResultsEvaluator():
    def __init__(self, EsN0: list, nMC:int):
        self.EsN0_dB = EsN0
        self.nMC = nMC
        self.NMSE_dB = np.zeros(len(EsN0))
        self.EVM_dB = np.zeros(len(EsN0))
        self.PreLDPCCodedBER = np.zeros(len(EsN0))
        self.TransportBlockBER = np.zeros(len(EsN0))
        self.BLER = np.zeros(len(EsN0))

        self.channelEstimation_error_power = np.zeros((len(EsN0), nMC), dtype=np.float64)
        self.channel_power = np.zeros((len(EsN0), nMC), dtype=np.float64)
        self.EVM_error_power = np.zeros((len(EsN0), nMC), dtype=np.float64)
        self.EVM_signal_power = np.zeros((len(EsN0), nMC), dtype=np.float64)
        self.CodedBER = np.zeros((len(EsN0), nMC), dtype=np.float64)
        self.TB_BER = np.zeros((len(EsN0), nMC), dtype=np.float64)
        self.BLER_ = np.zeros((len(EsN0), nMC), dtype=np.float64)
    
    def save_results(self, results: dict, EsnN0_idx: int, m: int):
        
        self.channelEstimation_error_power[EsnN0_idx, m] = results["error_power"]
        self.channel_power[EsnN0_idx, m] = results["channel_power"]
        self.EVM_error_power[EsnN0_idx, m] = results["evm_error_power"]
        self.EVM_signal_power[EsnN0_idx, m] = results["evm_signal_power"]
        self.CodedBER[EsnN0_idx, m] = results["coded_ber"]
        self.TB_BER[EsnN0_idx, m] = results["tb_ber"]
        self.BLER_[EsnN0_idx, m] = results["bler"]

    def process(self, EsN0_idx: int):
            
        numerator = np.sum(self.channelEstimation_error_power[EsN0_idx])
        denumerator = np.sum(self.channel_power[EsN0_idx])
        self.NMSE_dB[EsN0_idx] = 10 * np.log10(numerator / denumerator)

        numerator = np.sum(self.EVM_error_power[EsN0_idx])
        denumerator = np.sum(self.EVM_signal_power[EsN0_idx])
        evm = np.sqrt(numerator / denumerator)
        self.EVM_dB[EsN0_idx] = 20 * np.log10(evm)

        self.PreLDPCCodedBER[EsN0_idx] = np.mean(self.CodedBER[EsN0_idx]) * 100
        self.TransportBlockBER[EsN0_idx] = np.mean(self.TB_BER[EsN0_idx]) * 100
        self.BLER[EsN0_idx] = np.mean(self.BLER_[EsN0_idx]) * 100

        logging.info(f"-- NMSE: {self.NMSE_dB[EsN0_idx]:.2f} dB")
        logging.info(f"-- EVM: {self.EVM_dB[EsN0_idx]:.2f} dB")
        logging.info(f"-- Coded BER: {self.PreLDPCCodedBER[EsN0_idx]:.2f} %")
        logging.info(f"-- TB BER: {self.TransportBlockBER[EsN0_idx]:.2f} %")
        logging.info(f"-- BLER: {self.BLER[EsN0_idx]:.2f}%\n")
    
    def logging_overall_results(self):
        logging.info("===== Overall Summary =====")
        for i,EsN0 in enumerate(self.EsN0_dB):
            logging.info(f"SNR {EsN0} dB:")
            logging.info(f"   NMSE {self.NMSE_dB[i]:.2f} dB")
            logging.info(f"   EVM {self.EVM_dB[i]:.2f} dB")
            logging.info(f"   Coded BER {self.PreLDPCCodedBER[i]:.2f} %")
            logging.info(f"   TB BER {self.TransportBlockBER[i]:.2f} %")
            logging.info(f"   BLER {self.BLER[i]:.2f}%\n")

class Evaluator():
    def __init__(self, allocatedPDSCHSymbols, allocatedPRB):
        self.ChannelEstimationNMSE = ChannelEstimationNMSE(allocatedPDSCHSymbols, allocatedPRB)
        self.EVMCalculator = EVMCalculator()
        self.CodedBERCalculator = CodedBERCalculator()
        self.TB_BERCalculator = TB_BERCalculator()
        self.BLERCalculator = BLERCalculator()

    def process(self, Estimated: dict, GroundTruth: dict) -> dict:
        error_power, channel_power = self.ChannelEstimationNMSE.process(Estimated["Channel"], GroundTruth["Channel"])
        evm_error_power, evm_signal_power = self.EVMCalculator.process(Estimated["QAMSymbols"], GroundTruth["QAMSymbols"])
        coded_ber = self.CodedBERCalculator.process(Estimated["LLRs"], GroundTruth["ScrambledBits"])
        tb_ber = self.TB_BERCalculator.process(Estimated["TransportBlock"], GroundTruth["TransportBlock"])
        bler = self.BLERCalculator.process(Estimated["TransportBlock"], GroundTruth["TransportBlock"])

        results = {
            "error_power": error_power,
            "channel_power": channel_power,
            "evm_error_power": evm_error_power,
            "evm_signal_power": evm_signal_power,
            "coded_ber": coded_ber,
            "tb_ber": tb_ber,
            "bler": bler
        }
        return results
    
class BLERCalculator():
    def process(self, EstimatedTransportBlock: np.ndarray, GroundTruthTransportBlock: np.ndarray) -> int:
        bit_errors = np.count_nonzero(GroundTruthTransportBlock != EstimatedTransportBlock)
        isFail = bit_errors != 0
        return int(isFail)
    
class TB_BERCalculator():
    def process(self, EstimatedTransportBlock: np.ndarray, GroundTruthTransportBlock: np.ndarray) -> float:
        tb_ber = np.mean(GroundTruthTransportBlock != EstimatedTransportBlock)
        return tb_ber
    
class CodedBERCalculator():
    def process(self, LLRs: list[np.ndarray], ScrambledBits: list[np.ndarray]):
        hardDecisionLLRs = (LLRs[0] < 0).astype(np.uint8)
        coded_ber = np.mean(hardDecisionLLRs != ScrambledBits[0])
        return coded_ber

class EVMCalculator():
    def process(self, EstimatedQAMSymbols: list[np.ndarray], QAMSymbols: list[np.ndarray]):
        evm_error_power = np.sum(np.abs(EstimatedQAMSymbols[0] - QAMSymbols[0])**2)
        evm_signal_power = np.sum(np.abs(QAMSymbols[0])**2)
        return evm_error_power, evm_signal_power

class ChannelEstimationNMSE():
    def __init__(self, allocatedPDSCHSymbols: list, allocatedPRB: list):
        self.allocatedPDSCHSymbols = allocatedPDSCHSymbols
        allocatedSubcarriers = []
        for prb in allocatedPRB:
            start_sc = prb * 12
            end_sc = start_sc + 12
            allocatedSubcarriers.extend(range(start_sc, end_sc))
        self.allocatedSubcarriers = allocatedSubcarriers

    def process(self, EstimatedChannel: np.ndarray, GroundTruthChannel: np.ndarray) -> tuple[float, float]:
        nActiveSubcarrier = EstimatedChannel.shape[0]
        NFFT = GroundTruthChannel.shape[0]
        
        start_k = NFFT // 2 - nActiveSubcarrier // 2
        last_k = start_k + nActiveSubcarrier
        BWPChannel = GroundTruthChannel[start_k:last_k, :]

        H_gt = BWPChannel[self.allocatedSubcarriers,:][:, self.allocatedPDSCHSymbols]
        H_est = EstimatedChannel[self.allocatedSubcarriers,:][:, self.allocatedPDSCHSymbols]

        error = H_est - H_gt

        error_power = float(np.sum(np.abs(error) ** 2))
        channel_power = float(np.sum(np.abs(H_gt) ** 2))

        if channel_power == 0:
            raise ValueError(
                "Ground-truth channel power is zero."
            )

        return error_power, channel_power