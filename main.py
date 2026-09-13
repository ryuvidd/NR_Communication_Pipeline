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
        maxIter = 10
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
    logging_level(DEBUG_MODE)
    rng = np.random.default_rng()
    EsN0_dB = list(range(-5,20,5))
    nMC = 100
    HARQ_number = 0

    ThisSimulator = Simulator(TxConfig, RxConfig, ChannelConfig)
    Results = np.zeros((len(EsN0_dB),nMC), dtype=bool)
    SuccessRate = np.zeros(len(EsN0_dB))
    NMSE_dB = np.zeros(len(EsN0_dB))
    EVM = np.zeros(len(EsN0_dB))
    CodedBER = np.zeros(len(EsN0_dB))

    for i,EsN0 in enumerate(EsN0_dB):
        logging.info(f"===== Simulation under Es/N0 {EsN0} dB =====")
        total_error_power = 0
        total_channel_power = 0
        totalEVM = 0
        totalCodedBER = 0

        for m in range(nMC):
            results = ThisSimulator.process(DEBUG_MODE, EsN0, HARQ_number)
            total_error_power += results["error_power"]
            total_channel_power += results["channel_power"]
            totalEVM += results["evm"]
            totalCodedBER += results["coded_ber"]

            if results["isSuccess"]:
                logging.info(f"Transmission {m+1}: Success")
            else:
                logging.info(f"Transmission {m+1}: Failure")
            Results[i,m] = results["isSuccess"]

        NMSE_dB[i] = 10 * np.log10((total_error_power / total_channel_power))
        EVM[i] = totalEVM / nMC * 100
        CodedBER[i] = totalCodedBER / nMC * 100
        SuccessRate[i] = np.mean(Results[i]) * 100
        logging.info(f"-- NMSE: {NMSE_dB[i]:.3f} dB")
        logging.info(f"-- EVM: {EVM[i]:.3f} %")
        logging.info(f"-- Coded BER: {CodedBER[i]:.3f} %")
        logging.info(f"-- Success rate: {SuccessRate[i]:.3f}%\n")
    
    logging.info("===== Overall Summary =====")
    for i,EsN0 in enumerate(EsN0_dB):
        logging.info(f"SNR {EsN0} dB:")
        logging.info(f"   NMSE {NMSE_dB[i]:.3f} dB")
        logging.info(f"   EVM {EVM[i]:.3f} %")
        logging.info(f"   Coded BER {CodedBER[i]:.3f} %")
        logging.info(f"   Success rate {SuccessRate[i]:.3f}%\n")
    
