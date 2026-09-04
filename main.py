from Transmitter import *
from Receiver import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        n_SCID = 0
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
        n_SCID = 0
    )
    
    rng = np.random.default_rng(34)
    InformationData = rng.integers(0, 2, size=100000, dtype=np.uint8)

    ThisTransmitter = Transmitter(TxConfig)
    TransmittedWaveForm = ThisTransmitter.process(InformationData)
    logging.info("...................................")
    logging.info("...... Transmitted Wave Form ......")
    logging.info("...................................")
    ThisReceiver = Receiver(RxConfig)
    retransmissionCodeBlockIndices, EstimatedTransportBlock = ThisReceiver.process(TransmittedWaveForm)
    if len(retransmissionCodeBlockIndices) == 0:
        TransportBlock = InformationData[:ThisTransmitter.meta["TBS"]]
        if np.allclose(TransportBlock, EstimatedTransportBlock):
            logging.info("===== Success =====")
        else: logging.info("===== Failure =====")
    else: 
        logging.info("===== Failure =====")
    

    
