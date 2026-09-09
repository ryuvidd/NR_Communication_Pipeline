from Transmitter import *
from Receiver import *
from Channel import *
from util import *

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
        NFFT = 1024
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
    
    rng = np.random.default_rng(34)
    InformationData = rng.integers(0, 2, size=100000, dtype=np.uint8)

    ThisTransmitter = Transmitter(TxConfig)
    TransmittedSymbols = ThisTransmitter.process(InformationData)
    logging.info("...................................")
    logging.info("...... Transmitted Wave Form ......")
    Channel = RayleighFadingChannel(ChannelConfig)
    ChannelOutputs = Channel.process(TransmittedSymbols)
    logging.info("...................................")
    ThisReceiver = Receiver(RxConfig)
    retransmissionCodeBlockIndices, EstimatedTransportBlock = ThisReceiver.process(ChannelOutputs)
    
    if len(retransmissionCodeBlockIndices) == 0:
        TransportBlock = InformationData[:ThisTransmitter.meta["TBS"]]
        if np.allclose(TransportBlock, EstimatedTransportBlock):
            logging.info("===== Success =====")
        else: logging.info("===== Failure =====")
    else: 
        logging.info("===== Failure =====")
    

    
