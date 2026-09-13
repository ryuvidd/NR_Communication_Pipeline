import numpy as np

class Evaluator():
    def __init__(self, allocatedPDSCHSymbols, allocatedPRB):
        self.ChannelEstimationNMSE = ChannelEstimationNMSE(allocatedPDSCHSymbols, allocatedPRB)
        self.EVMCalculator = EVMCalculator()
        self.CodedBERCalculator = CodedBERCalculator()
        self.TB_BERCalculator = TB_BERCalculator()
        self.BLERCalculator = BLERCalculator()

    def process(self, Estimated: dict, GroundTruth: dict) -> dict:
        error_power, channel_power = self.ChannelEstimationNMSE.process(Estimated["Channel"], GroundTruth["Channel"])
        evm = self.EVMCalculator.process(Estimated["QAMSymbols"], GroundTruth["QAMSymbols"])
        coded_ber = self.CodedBERCalculator.process(Estimated["LLRs"], GroundTruth["ScrambledBits"])
        tb_ber = self.TB_BERCalculator.process(Estimated["TransportBlock"], GroundTruth["TransportBlock"])
        bler = self.BLERCalculator.process(Estimated["TransportBlock"], GroundTruth["TransportBlock"])

        results = {
            "error_power": error_power,
            "channel_power": channel_power,
            "evm": evm,
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
        evm = np.sqrt(np.sum(np.abs(EstimatedQAMSymbols[0] - QAMSymbols[0])**2) / np.sum(np.abs(QAMSymbols[0])**2))
        return evm

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