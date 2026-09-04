import numpy as np
from dataclasses import dataclass

T_C = 1 / (480e3 * 4096)

@dataclass
class OFDMModulationConfig:
    nPRB: int
    nOFDMSymbolsPerSlot: int
    SubcarrierSpacing: float
    NFFT: int

class OFDMModulator:

    def __init__(self, config: OFDMModulationConfig):
        self.nPRB = config.nPRB
        self.N_SC = config.nPRB * 12
        self.L = config.nOFDMSymbolsPerSlot
        self.delta_f = config.SubcarrierSpacing
        self.NFFT = config.NFFT
        self.mu = int(np.log2(self.delta_f / 15e3))
        self.Fs = self.NFFT * self.delta_f
        self.N_CP = self.__compute_N_CP__()

    def __compute_N_CP__(self):

        if self.L not in (12, 14):
            raise ValueError("Number of OFDM symbols per slot must be 12 or 14.")

        if self.L == 14:
            cp_normal_time = (144 * 64 / (2 ** self.mu)) * T_C
            cp_first_time = ((144 * 64 / (2 ** self.mu)) + 16 * 64) * T_C

            cp_normal_samples = round(cp_normal_time * self.Fs)
            cp_first_samples = round(cp_first_time * self.Fs)

            N_CP = [cp_first_samples] + [cp_normal_samples] * (self.L - 1)
        else:
            cp_time = (512 * 64 / (2 ** self.mu)) * T_C
            cp_samples = round(cp_time * self.Fs)
            N_CP = [cp_samples] * self.L

        return N_CP

    def process(self, grid: np.ndarray) -> np.ndarray:
        ## following TS38.211 clause 5.3.1, assuming k0 is zero ##

        if grid.shape != (self.N_SC, self.L):
            raise ValueError(f"Expected grid shape ({self.N_SC}, {self.L}), got {grid.shape}")

        waveForm = []

        for l in range(self.L):

            # handling - N_SC/2 by mapping active subcarriers to FFT bins centered around DC
            extendedSubcarrier = np.zeros(self.NFFT, dtype=complex)
            start = (self.NFFT // 2 - self.N_SC // 2)
            extendedSubcarrier[start:start + self.N_SC] = grid[:, l]
            centeredSubcarrier = np.fft.ifftshift(extendedSubcarrier)

            # - N_CP * T_C is used in equation for specifying the useful samples and t_start is handled by concatenation.
            ModulatedSamples = np.fft.ifft(centeredSubcarrier)
            cp_length_samples = self.N_CP[l]
            InsertedCP = np.concatenate([ModulatedSamples[-cp_length_samples:], ModulatedSamples])
            waveForm.append(InsertedCP)
        SampledTransmittedWave = np.concatenate(waveForm)
        return SampledTransmittedWave

class OFDMDemodulator:

    def __init__(self, config: OFDMModulationConfig):
        self.nPRB = config.nPRB
        self.N_SC = config.nPRB * 12
        self.L = config.nOFDMSymbolsPerSlot
        self.delta_f = config.SubcarrierSpacing
        self.NFFT = config.NFFT
        self.mu = int(np.log2(self.delta_f / 15e3))
        self.Fs = self.NFFT * self.delta_f
        self.N_CP = self.__compute_N_CP__()

    def __compute_N_CP__(self):

        if self.L not in (12, 14):
            raise ValueError("Number of OFDM symbols per slot must be 12 or 14.")

        if self.L == 14:
            cp_normal_time = (144 * 64 / (2 ** self.mu)) * T_C
            cp_first_time = ((144 * 64 / (2 ** self.mu)) + 16 * 64) * T_C

            cp_normal_samples = round(cp_normal_time * self.Fs)
            cp_first_samples = round(cp_first_time * self.Fs)

            N_CP = [cp_first_samples] + [cp_normal_samples] * (self.L - 1)
        else:
            cp_time = (512 * 64 / (2 ** self.mu)) * T_C
            cp_samples = round(cp_time * self.Fs)
            N_CP = [cp_samples] * self.L

        return N_CP
    
    def process(self, ReceivedSignal: np.ndarray) -> np.ndarray:
        EstimatedGrid = np.zeros((self.N_SC, self.L), dtype=complex)

        start_sample = 0
        for l in range(self.L):
            cp_length_samples = self.N_CP[l]
            symbol_length = cp_length_samples + self.NFFT
            receivedSymbol = ReceivedSignal[start_sample:start_sample + symbol_length]
            usefulSymbol = receivedSymbol[cp_length_samples:]

            X_hat = np.fft.fft(usefulSymbol)
            X_centered = np.fft.fftshift(X_hat)
            start_bin = (self.NFFT // 2 - self.N_SC // 2)
            EstimatedGrid[:, l] = X_centered[start_bin: start_bin + self.N_SC]
            start_sample += symbol_length
        return EstimatedGrid

if __name__ == "__main__":
    config = OFDMModulationConfig(
        nPRB = 10,
        nOFDMSymbolsPerSlot = 14,
        SubcarrierSpacing = int(15e3),
        NFFT = 1024
    )

    grid = np.random.rand(config.nPRB*12, config.nOFDMSymbolsPerSlot) + 1j*np.random.rand(config.nPRB*12, config.nOFDMSymbolsPerSlot)
    ThisOFDMModulator = OFDMModulator(config)
    TxWave = ThisOFDMModulator.process(grid)

    ThisOFDMDemodulator = OFDMDemodulator(config)
    EstimatedGrid = ThisOFDMDemodulator.process(TxWave)
    print("Maximum absolute difference: ", np.max(np.abs(grid - EstimatedGrid)))
    print("No significant error: ", np.allclose(grid, EstimatedGrid))
    