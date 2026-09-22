import numpy as np
from dataclasses import dataclass
from enum import Enum
from Configuration import *
    
@dataclass
class RayleighFadingConfig:
    velocity: float
    carrierFrequency: float
    delays_ns: list
    delayPower_dB: list

def select_channel_model(channel_config: ChannelConfig):
    if channel_config.model == CHANNEL_MODEL.Rayleigh:
        temp_config =  RayleighFadingConfig(
            velocity=channel_config.velocity,
            carrierFrequency=channel_config.carrierFrequency,
            delays_ns=channel_config.delays_ns,
            delayPower_dB=channel_config.delayPower_dB
        )
        return RayleighFadingChannel(temp_config)
    else:
        raise ValueError("Only support Rayleigh's fading channel for now.")

class RayleighFadingChannel():
    def __init__(self, config:RayleighFadingConfig):

        self.velocity = config.velocity
        self.fc = config.carrierFrequency

        self.delays_ns = np.asarray(config.delays_ns)
        self.nTap = len(self.delays_ns)

        power = 10 ** (np.asarray(config.delayPower_dB) / 10)
        self.power_linear = power / np.sum(power)

        self.already_got_stat = False

    def compare_coherence_time(self, Ts: float, subcarreierSpacing: float):
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

    def generate_channels(self, L: int, NFFT: int) -> tuple[list[np.ndarray], np.ndarray]:
        channels = []
        channelFrequency = []

        nRealization = 1 if self.ConstantOver == "SLOT" else L
        maxDelay = np.max(self.delays_sample)

        for _ in range(nRealization):

            h = np.sqrt(0.5) * (np.random.randn(self.nTap) + 1j * np.random.randn(self.nTap))
            channel = np.zeros(maxDelay + 1, dtype=np.complex128)
            for tap, delay in enumerate(self.delays_sample):
                channel[delay] += h[tap] * np.sqrt(self.power_linear[tap])
            channels.append(channel)

            channelFrequency.append(np.fft.fftshift(np.fft.fft(channel, n=NFFT)))

        if self.ConstantOver == "SLOT":
            channels = [channels[0]] * L
            channelFrequency = [channelFrequency[0]] * L

        channelFrequency = np.array(channelFrequency).T
        return channels, channelFrequency
    
    def get_stat_info(self, L: int, NFFT: int, nMonteCarlo:int):
        np.random.seed(10)
        ChannelFreq = np.zeros((nMonteCarlo, NFFT), dtype=np.complex128)
        for m in range(nMonteCarlo):
            _, ThisChannelFreq = self.generate_channels(L, NFFT)
            ChannelFreq[m] = ThisChannelFreq[:,0]

        self.mean_h = np.mean(ChannelFreq, axis=0)
        R_hh = np.zeros((NFFT, NFFT), dtype=np.complex128)
        for m in range(nMonteCarlo):
            temp = (ChannelFreq[m] - self.mean_h).reshape(-1,1)
            R_hh += temp @ temp.conj().T
        self.R_hh = R_hh / (nMonteCarlo - 1)

        self.already_got_stat = True
        

    def process(self, channel_input: dict):
        TransmittedSymbols = channel_input["TransmittedSymbols"]
        Ts = channel_input["Ts"]
        subcarrierSpacing = channel_input["subcarrierSpacing"]
        NFFT = channel_input["NFFT"]
        L = len(TransmittedSymbols)

        self.ConstantOver, self.delays_sample = self.compare_coherence_time(Ts, subcarrierSpacing)
        if not self.already_got_stat:
            self.get_stat_info(L, NFFT, nMonteCarlo=1000)

        self.Channels, self.ChannelFrequency = self.generate_channels(L, NFFT)
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

if __name__ == '__main__':

    ChannelConfig_ = RayleighFadingConfig(
        velocity = 15,
        carrierFrequency = 2.5e9,
        delays_ns = [0, 100, 300],
        delayPower_dB = [0, -3, -6]
    )

    TxWave = [np.arange(100)] * 14
    T_slot = 2e-3
    Ts = 3.2e-8
    NFFT = 1024
    delta_f = 2.5e9

    Channel = RayleighFadingChannel(ChannelConfig_)
    channel_input = {
        "TransmittedSymbols": TxWave,
        "Ts": Ts,
        "subcarrierSpacing": delta_f,
        "NFFT": NFFT
    }
    ChannelOutputs = Channel.process(channel_input)