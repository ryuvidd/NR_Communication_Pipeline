import logging
import matplotlib.pyplot as plt
import pickle

def logging_level(DEBUG_MODE: bool):
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def plot_results(results: dict, save_name:str):
    EsN0_dB = results["EsN0_dB"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # NMSE
    axes[0, 0].plot(EsN0_dB, results["NMSE_dB"], marker='o')
    axes[0, 0].set_xlabel("Es/N0 (dB)")
    axes[0, 0].set_ylabel("NMSE (dB)")
    axes[0, 0].set_title("NMSE")
    axes[0, 0].grid(True)

    # EVM
    axes[0, 1].plot(EsN0_dB, results["EVM_dB"], marker='o')
    axes[0, 1].set_xlabel("Es/N0 (dB)")
    axes[0, 1].set_ylabel("EVM (dB)")
    axes[0, 1].set_title("EVM")
    axes[0, 1].grid(True)

    # Coded BER
    axes[1, 0].plot(EsN0_dB, results["CodedBER"], marker='o')
    axes[1, 0].set_xlabel("Es/N0 (dB)")
    axes[1, 0].set_ylabel("Pre-LDPC Coded BER (%)")
    axes[1, 0].set_title("Pre-LDPC Coded BER")
    axes[1, 0].grid(True)

    # Transport Block BER
    axes[1, 1].plot(EsN0_dB, results["TB_BER"], marker='o')
    axes[1, 1].set_xlabel("Es/N0 (dB)")
    axes[1, 1].set_ylabel("Transport Block BER (%)")
    axes[1, 1].set_title("Transport Block BER")
    axes[1, 1].grid(True)

    # BLER
    axes[2, 0].plot(EsN0_dB, results["BLER"], marker='o')
    axes[2, 0].set_xlabel("Es/N0 (dB)")
    axes[2, 0].set_ylabel("BLER (%)")
    axes[2, 0].set_title("BLER")
    axes[2, 0].grid(True)

    # Leave the last subplot blank
    axes[2, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_name, dpi=400)

def plot_all_results(load_results_name: list[str], save_name: str):
    
    ALL_NMSE_dB = []
    ALL_EVM_dB = []
    ALL_CodedBER = []
    ALL_TB_BER = []
    ALL_BLER = []
    simulation_labels = ["LS, ZF", "LMMSE, ZF", "LS, MMSE", "LMMSE, MMSE"]
    nSimulations = len(simulation_labels)
    
    for i in range(nSimulations):
        with open(load_results_name[i], "rb") as f:
            results = pickle.load(f)

        EsN0_dB = results["EsN0_dB"]
        ALL_NMSE_dB.append(results["NMSE_dB"])
        ALL_EVM_dB.append(results["EVM_dB"])
        ALL_CodedBER.append(results["CodedBER"])
        ALL_TB_BER.append(results["TB_BER"])
        ALL_BLER.append(results["BLER"])

    ALL_RESULTS = [ALL_NMSE_dB, ALL_EVM_dB, ALL_CodedBER, ALL_TB_BER, ALL_BLER]
    
    metric_labels = ["NMSE (dB)", "EVM (dB)", "Pre-LDPC Coded BER (%)", "Transport Block BER (%)", "BLER (%)"]
    subplot_titles = ["Channel Estimation NMSE", "EVM", "Pre-LDPC Coded BER", "Transport Block BER", "BLER"]
    
    metric_idx = 0
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for i in range(3):
        for j in range(2):
            if (i,j) == (2,1):
                axes[i,j].axis("off")
            else:
                for k in range(nSimulations):
                    axes[i,j].plot(EsN0_dB, ALL_RESULTS[metric_idx][k], marker='o', label=simulation_labels[k])
                axes[i,j].set_xlabel("Es/N0 (dB)")
                axes[i,j].set_ylabel(metric_labels[metric_idx])
                axes[i,j].set_title(subplot_titles[metric_idx])
                axes[i,j].grid(True)
                axes[i,j].legend(simulation_labels, loc="upper right")
                metric_idx += 1        

    plt.tight_layout()
    plt.savefig(save_name, dpi=400)

if __name__ == "__main__":
    
    load_results_name = [
        "results/LS_ZF.pkl",
        "results/LMMSE_ZF.pkl",
        "results/LS_MMSE.pkl",
        "results/LMMSE_MMSE.pkl"
    ]

    save_name = "figs/Overall_Results.png"
    plot_all_results(load_results_name, save_name)