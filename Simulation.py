from Transmitter import *
from Receiver import *
from Channel import *
from util import *

class Simulator():
    def __init__(self, TxConfig: TransmitterConfig, RxConfig: ReceiverConfig, ChannelConfig: RayleighFadingConfig):
        self.Transmitter = Transmitter(TxConfig)
        self.Channel = RayleighFadingChannel(ChannelConfig)
        self.NoiseMixer = NoiseMixer()
        self.Receiver = Receiver(RxConfig)

    def process(self, DEBUG_MODE: bool, SNR: float):
        logging_level(DEBUG_MODE)
        rng = np.random.default_rng()

        InformationData = rng.integers(0, 2, size=100000, dtype=np.uint8)
        TransmittedSymbols = self.Transmitter.process(InformationData)
        logging.debug("...................................")
        logging.debug("...... Transmitted Wave Form ......")
        ChannelOutputs = self.Channel.process(TransmittedSymbols)
        ReceivedSignal, VarNoise = self.NoiseMixer.process(ChannelOutputs, SNR)
        logging.debug("...................................")
        retransmissionCodeBlockIndices, EstimatedTransportBlock = self.Receiver.process(ReceivedSignal, VarNoise)
        
        isSuccess = False
        if len(retransmissionCodeBlockIndices) == 0:
            TransportBlock = InformationData[:self.Transmitter.meta["TBS"]]
            if np.allclose(TransportBlock, EstimatedTransportBlock):
                isSuccess = True
        
        return isSuccess

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
        nOFDMSymbolsPerSlot = 14
    )
    
    DEBUG_MODE = False
    SNR = 0

    ThisSimulator = Simulator(TxConfig, RxConfig, ChannelConfig)
    isSuccess = ThisSimulator.process(DEBUG_MODE, SNR)
    if isSuccess:
        logging.info(f"===== Transmission: Success =====")
    else:
        logging.info(f"===== Transmission: Failure =====")
