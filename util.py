import logging
import matplotlib.pyplot as plt

def logging_level(DEBUG_MODE: bool):
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def plot_results(EsN0_dB: list, results: dict, save_name:str):
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