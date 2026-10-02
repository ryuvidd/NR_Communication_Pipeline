from Simulation import *
from EvaluationMetrics import *
import pickle

if __name__ == '__main__':   

    config01 = NRSystemConfig(
        carrier=carrierConfig,
        bwp=bwpConfig,
        pdsch=pdschConfig,
        dmrs=dmrsConfig,
        scrambling=scramblingConfig,
        harq=harqConfig,
        channel=channelConfig,
        receiver=receiver01Config,
    )
    config02 = NRSystemConfig(
        carrier=carrierConfig,
        bwp=bwpConfig,
        pdsch=pdschConfig,
        dmrs=dmrsConfig,
        scrambling=scramblingConfig,
        harq=harqConfig,
        channel=channelConfig,
        receiver=receiver02Config,
    )
    config03 = NRSystemConfig(
        carrier=carrierConfig,
        bwp=bwpConfig,
        pdsch=pdschConfig,
        dmrs=dmrsConfig,
        scrambling=scramblingConfig,
        harq=harqConfig,
        channel=channelConfig,
        receiver=receiver03Config,
    )
    config04 = NRSystemConfig(
        carrier=carrierConfig,
        bwp=bwpConfig,
        pdsch=pdschConfig,
        dmrs=dmrsConfig,
        scrambling=scramblingConfig,
        harq=harqConfig,
        channel=channelConfig,
        receiver=receiver04Config,
    )

    DEBUG_MODE = False
    logging_level(DEBUG_MODE)
    np.random.seed(45)
    EsN0_dB = list(range(-4,11,2))
    nMC = 1000
    HARQ_number = 0
    compare_RVs_flag = True     # Simulate LMMSE-MMSE with all RV ids

    if compare_RVs_flag:
        numVariations = 4
        configs = [config04]
        save_results_name = []
        for k in range(numVariations):
            save_results_name.append("results/LMMSE_MMSE_RV_" + str(k+1) + "_Transmission.pkl")    
        
    else:
        configs = [config01, config02, config03, config04]
        numVariations = len(configs)
        save_results_name = []
        for k in range(len(configs)):
            save_results_name.append("results/" + configs[k].receiver.channelEstimatorType.value + "_" + configs[k].receiver.equalizerType.value + ".pkl")    

    ThisSimulator = Simulator(configs, compare_RVs_flag)
    ThisResultsEvaluator = ResultsEvaluator(EsN0_dB, nMC, numVariations, compare_RVs_flag)
    logging.info("")
    logging.info(f"***** Initialize Simulations *****\n")

    for i,EsN0 in enumerate(EsN0_dB):
        logging.info(f"===== Simulation under Es/N0 {EsN0} dB =====")
        for m in range(nMC):
            results = ThisSimulator.process(DEBUG_MODE, EsN0, HARQ_number)
            ThisResultsEvaluator.save_results(results, i, m)
            if ((m+1) % 10) == 0: 
                logging.info(f"Es/N0 {EsN0} dB - Transmission {m+1}: Done")

        ThisResultsEvaluator.process(i)
        
    ThisResultsEvaluator.logging_overall_results()

    if compare_RVs_flag:
        for k in range(numVariations):
            Plotting_results = {
                "EsN0_dB": EsN0_dB,
                "TB_BER": ThisResultsEvaluator.TransportBlockBER[k],
                "BLER": ThisResultsEvaluator.BLER[k],
                "TB_BER_": ThisResultsEvaluator.TB_BER[k],
                "BLER_": ThisResultsEvaluator.BLER_[k]
            }
        
            with open(save_results_name[k], "wb") as f:
                pickle.dump(Plotting_results, f)

        load_results_name = [
            "results/LMMSE_MMSE_RV_1_Transmission.pkl",
            "results/LMMSE_MMSE_RV_2_Transmission.pkl",
            "results/LMMSE_MMSE_RV_3_Transmission.pkl",
            "results/LMMSE_MMSE_RV_4_Transmission.pkl"
        ]

        save_fig_name = "figs/Results_LMMSE_MMSE_RVs.png"
        plot_RV_results(load_results_name, save_fig_name)

        logging.info("***** Simulation Completed *****\n")
    else:
        for k in range(numVariations):
            Plotting_results = {
                "EsN0_dB": EsN0_dB,
                "NMSE_dB": ThisResultsEvaluator.NMSE_dB[k],
                "EVM_dB": ThisResultsEvaluator.EVM_dB[k],
                "CodedBER": ThisResultsEvaluator.PreLDPCCodedBER[k],
                "TB_BER": ThisResultsEvaluator.TransportBlockBER[k],
                "BLER": ThisResultsEvaluator.BLER[k],
                "channelEstimation_error_power": ThisResultsEvaluator.channelEstimation_error_power[k],
                "channel_power": ThisResultsEvaluator.channel_power[k],
                "EVM_error_power": ThisResultsEvaluator.EVM_error_power[k],
                "EVM_signal_power": ThisResultsEvaluator.EVM_signal_power[k],
                "CodedBER_": ThisResultsEvaluator.CodedBER[k],
                "TB_BER_": ThisResultsEvaluator.TB_BER[k],
                "BLER_": ThisResultsEvaluator.BLER_[k]
            }
        
            with open(save_results_name[k], "wb") as f:
                pickle.dump(Plotting_results, f)
    
        load_results_name = [
            # "results/LS_ZF.pkl",
            # "results/LMMSE_ZF.pkl",
            "results/LS_MMSE.pkl",
            "results/LMMSE_MMSE.pkl"
        ]

        save_fig_name = "figs/Results_LS_LMMSE_One_Transmission.png"
        plot_all_results(load_results_name, save_fig_name)

        logging.info("***** Simulation Completed *****\n")
