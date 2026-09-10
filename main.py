from Simulation import *

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
    logging_level(DEBUG_MODE)
    rng = np.random.default_rng()
    SNR = list(range(-4,7,2))
    nMC = 2

    ThisSimulator = Simulator(TxConfig, RxConfig, ChannelConfig)
    Results = np.zeros((len(SNR),nMC), dtype=bool)
    SuccessRate = np.zeros(len(SNR))
    for i,snr in enumerate(SNR):
        logging.info(f"===== Simulation under SNR {snr}dB =====")
        for m in range(nMC):
            isSuccess = ThisSimulator.process(DEBUG_MODE, snr)
            if isSuccess:
                logging.info(f"Transmission {m+1}: Success")
            else:
                logging.info(f"Transmission {m+1}: Failure")
            Results[i,m] = isSuccess
        SuccessRate[i] = np.mean(Results[i]) * 100
        logging.info(f"-- Success rate: {SuccessRate[i]:.3f}%\n")
    
    logging.info("===== Overall Summary =====")
    for i,snr in enumerate(SNR):
        logging.info(f"SNR {snr}dB: {SuccessRate[i]:.3f}% success")
    
