from Simulation import *
from EvaluationMetrics import *

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
    logging_level(DEBUG_MODE)
    rng = np.random.default_rng()
    EsN0_dB = list(range(-4,11,2))
    nMC = 2
    HARQ_number = 0

    ThisSimulator = Simulator(config)
    ThisResultsEvaluator = ResultsEvaluator(EsN0_dB, nMC)

    for i,EsN0 in enumerate(EsN0_dB):
        logging.info(f"===== Simulation under Es/N0 {EsN0} dB =====")
        for m in range(nMC):
            results = ThisSimulator.process(DEBUG_MODE, EsN0, HARQ_number)
            ThisResultsEvaluator.save_results(results, i, m)

            if results["bler"] == 0:
                logging.info(f"Transmission {m+1}: Success")
            else:
                logging.info(f"Transmission {m+1}: Failure")

        ThisResultsEvaluator.process(i)
        
    ThisResultsEvaluator.logging_overall_results()

    Plotting_results = {
        "NMSE_dB": ThisResultsEvaluator.NMSE_dB,
        "EVM_dB": ThisResultsEvaluator.EVM_dB,
        "CodedBER": ThisResultsEvaluator.PreLDPCCodedBER,
        "TB_BER": ThisResultsEvaluator.TransportBlockBER,
        "BLER": ThisResultsEvaluator.BLER
    }
    save_fig_name = "figs/Results.png"
    plot_results(EsN0_dB, Plotting_results, save_fig_name)
    
