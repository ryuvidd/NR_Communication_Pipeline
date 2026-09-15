import numpy as np
from dataclasses import dataclass
from enum import Enum
from Configuration import *
    
@dataclass
class RayleighFadingConfig2:
    velocity: float
    carrierFrequency: float
    delays_ns: list
    delayPower_dB: list

def select_channel_model(channel_config: ChannelConfig):
    if channel_config.model == CHANNEL_MODEL.Rayleigh:
        temp_config =  RayleighFadingConfig2(
            velocity=channel_config.velocity,
            carrierFrequency=channel_config.carrierFrequency,
            delays_ns=channel_config.delays_ns,
            delayPower_dB=channel_config.delayPower_dB
        )
        return RayleighFadingChannel2(temp_config)
    else:
        raise ValueError("Only support Rayleigh's fading channel for now.")

class RayleighFadingChannel2():
    def __init__(self, config:RayleighFadingConfig2):

        self.velocity = config.velocity
        self.fc = config.carrierFrequency

        self.delays_ns = np.asarray(config.delays_ns)
        self.nTap = len(self.delays_ns)

        power = 10 ** (np.asarray(config.delayPower_dB) / 10)
        self.power_linear = power / np.sum(power)

    def compare_coherence_time(self, Ts, subcarreierSpacing):
        Doppler = self.velocity * self.fc / 299792458
        coherenceTime = 0.423 / Doppler

        T_symbol = 1 / subcarreierSpacing

        mu = np.log2(subcarreierSpacing / 15)
        T_slot = 1e-3 / (2 ** mu)

        if coherenceTime >= T_slot:
            ConstantOver = "SLOT"
        elif coherenceTime >= T_symbol:
            ConstantOver = "SYMBOL"
        else:
            ConstantOver = "SAMPLE"
            raise ValueError("Only support channel is constant over slot or symbol for now.")

        delays_sample = np.round(self.delays_ns * 1e-9 / Ts).astype(int)
        return ConstantOver, delays_sample

    def generate_channels(self, ConstantOver, L, delays_sample, NFFT) -> tuple[list[np.ndarray], np.ndarray]:
        channels = []
        channelFrequency = []

        nRealization = 1 if ConstantOver == "SLOT" else L
        maxDelay = np.max(delays_sample)

        for _ in range(nRealization):

            h = np.sqrt(0.5) * (np.random.randn(self.nTap) + 1j * np.random.randn(self.nTap))
            channel = np.zeros(maxDelay + 1, dtype=np.complex128)
            for tap, delay in enumerate(delays_sample):
                channel[delay] += h[tap] * np.sqrt(self.power_linear[tap])
            channels.append(channel)

            channelFrequency.append(np.fft.fftshift(np.fft.fft(channel, n=NFFT)))

        if ConstantOver == "SLOT":
            channels = [channels[0]] * L
            channelFrequency = [channelFrequency[0]] * L

        channelFrequency = np.array(channelFrequency).T
        return channels, channelFrequency

    def process(self, TransmittedSymbols: list[np.ndarray], Ts: float, subcarrierSpacing: float, NFFT):
        L = len(TransmittedSymbols)

        ConstantOver, delays_sample = self.compare_coherence_time(Ts, subcarrierSpacing)
        self.Channels, self.ChannelFrequency = self.generate_channels(ConstantOver, L, delays_sample, NFFT)
        ChannelOutputs = []
        for l,symbol in enumerate(TransmittedSymbols):
            ChannelOutputs.append(np.convolve(symbol, self.Channels[l]))
            
        return ChannelOutputs

class AWGNChannel():
    def __init__(self, NFFT: int, EsN0_dB: float):
        self.EsN0_dB = EsN0_dB
        self.NFFT = NFFT

    def process(self, ChannelOutputSymbols: list[np.ndarray]) -> tuple:
        Es = 1
        EsN0_linear = 10 ** (self.EsN0_dB / 10)
        N0 = Es / EsN0_linear
        noise_power = N0 / self.NFFT
        
        ChannelOutput = []
        for symbol in ChannelOutputSymbols:
            Noise = np.sqrt(noise_power / 2) * (np.random.randn(symbol.size) + 1j * np.random.randn(symbol.size))
            ChannelOutput.append(symbol + Noise)

        return ChannelOutput, N0
    
@dataclass
class RayleighFadingConfig:
    velocity: float
    carrierFrequency: float
    delays_ns: list
    delayPower_dB: list
    Ts: float
    T_slot: float
    nOFDMSymbolsPerSlot: int
    NFFT: int

class RayleighFadingChannel():
    def __init__(self, config:RayleighFadingConfig):

        self.fc = config.carrierFrequency
        self.velocity = config.velocity
        self.Ts = config.Ts
        self.T_slot = config.T_slot
        self.L = config.nOFDMSymbolsPerSlot
        self.T_symbol = self.T_slot / self.L
        self.NFFT = config.NFFT

        Doppler = self.velocity * self.fc / 299792458
        self.coherenceTime = 0.423 / Doppler

        if self.coherenceTime >= self.T_slot:
            self.ConstantOver = "SLOT"
        elif self.coherenceTime >= self.T_symbol:
            self.ConstantOver = "SYMBOL"
        else:
            self.ConstantOver = "SAMPLE"

        self.delays_ns = np.asarray(config.delays_ns)
        self.nTap = len(self.delays_ns)

        power = 10 ** (np.asarray(config.delayPower_dB) / 10)
        self.power_linear = power / np.sum(power)

        self.delays_sample = np.round(self.delays_ns * 1e-9 / self.Ts).astype(int)

        self.Channels, self.ChannelFrequency = self.__generate_channels__()

    def __generate_channels__(self) -> tuple[list[np.ndarray], np.ndarray]:
        channels = []
        channelFrequency = []

        nRealization = 1 if self.ConstantOver == "SLOT" else self.L
        maxDelay = np.max(self.delays_sample)

        for _ in range(nRealization):

            h = np.sqrt(0.5) * (np.random.randn(self.nTap) + 1j * np.random.randn(self.nTap))
            channel = np.zeros(maxDelay + 1, dtype=np.complex128)
            for tap, delay in enumerate(self.delays_sample):
                channel[delay] += h[tap] * np.sqrt(self.power_linear[tap])
            channels.append(channel)

            channelFrequency.append(np.fft.fftshift(np.fft.fft(channel, n=self.NFFT)))

        if self.ConstantOver == "SLOT":
            channels = [channels[0]] * self.L
            channelFrequency = [channelFrequency[0]] * self.L

        channelFrequency = np.array(channelFrequency).T
        return channels, channelFrequency

    def process(self, TransmittedSymbols: list[np.ndarray]):

        ChannelOutputs = []
        for l,symbol in enumerate(TransmittedSymbols):
            ChannelOutputs.append(np.convolve(symbol, self.Channels[l]))
            
        return ChannelOutputs

if __name__ == '__main__':

    ChannelConfigg = RayleighFadingConfig(
        velocity = 15,
        carrierFrequency = 2.5e9,
        delays_ns = [0, 100, 300],
        delayPower_dB = [0, -3, -6],
        Ts = 3.2e-8,
        T_slot = 1 / 30e-9,
        nOFDMSymbolsPerSlot = 14,
        NFFT = 1024
    )

    TxWave = [np.arange(100)] * 14
    T_slot = 2e-3

    Channel = RayleighFadingChannel(ChannelConfigg)
    ChannelOutputs = Channel.process(TxWave)