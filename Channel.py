import numpy as np
from dataclasses import dataclass

@dataclass
class RayleighFadingConfig:
    velocity: float
    carrierFrequency: float
    delays_ns: list
    delayPower_dB: list
    Ts: float
    T_slot: float
    nOFDMSymbolsPerSlot: int

class NoiseMixer():
    def process(self, ChannelOutputSymbols: list[np.ndarray], SNR: float) -> tuple:
        ChannelOutput = []
        SNRlinear = 10 ** (SNR / 10)

        SignalPower = np.zeros(len(ChannelOutputSymbols))
        for l,symbol in enumerate(ChannelOutputSymbols):
            SignalPower[l] = np.mean(np.abs(symbol) ** 2)
        SignalPower = np.mean(SignalPower[SignalPower > 0])
        NoisePower = SignalPower / SNRlinear
        VarNoise = NoisePower
        
        for l,symbol in enumerate(ChannelOutputSymbols):
            Noise = np.sqrt(NoisePower / 2) * (np.random.randn(symbol.size) + 1j * np.random.randn(symbol.size))
            self.Noise = Noise
            ChannelOutput.append(symbol + Noise)

        return ChannelOutput, VarNoise

class RayleighFadingChannel():
    def __init__(self, config:RayleighFadingConfig):

        self.fc = config.carrierFrequency
        self.velocity = config.velocity
        self.Ts = config.Ts
        self.T_slot = config.T_slot
        self.L = config.nOFDMSymbolsPerSlot
        self.T_symbol = self.T_slot / self.L

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

        self.Channels = self.__generate_channels__()

    def __generate_channels__(self) -> list[np.ndarray]:
        channels = []

        nRealization = 1 if self.ConstantOver == "SLOT" else self.L
        maxDelay = np.max(self.delays_sample)

        for _ in range(nRealization):

            h = np.sqrt(0.5) * (np.random.randn(self.nTap) + 1j * np.random.randn(self.nTap))
            channel = np.zeros(maxDelay + 1, dtype=np.complex128)
            for tap, delay in enumerate(self.delays_sample):
                channel[delay] = h[tap] * np.sqrt(self.power_linear[tap])
            channels.append(channel)

        if self.ConstantOver == "SLOT":
            channels = [channels[0]] * self.L

        return channels

    def process(self, TransmittedSymbols: list[np.ndarray]):

        ChannelOutputs = []
        for l,symbol in enumerate(TransmittedSymbols):
            ChannelOutputs.append(np.convolve(symbol, self.Channels[l]))
        return ChannelOutputs

            



        # Channels = []
        # ChannelOutputs = []
        # # NumChannelRealization = int(np.ceil(signal.shape[0]/self.RegenChannel))
        # for i in range(NumChannelRealization):
        #     channel = np.sqrt(1/2) * (np.random.randn(1,self.NumTap) + 1j * np.random.randn(1,self.NumTap))
        #     THISsignal = signal[i*self.RegenChannel:(i+1)*self.RegenChannel]
        #     Output = np.array([np.convolve(row, channel.reshape(-1), mode='full') for row in THISsignal])
        #     Channels.append(channel)
        #     ChannelOutputs.append(Output)
        # Channels = np.concatenate(Channels, axis=0)
        # # Channels = np.concat((Channels, np.zeros((NumChannelRealization,NumSubCarrier-self.NumTap))), axis=1)
        # # self.Channels_feq = np.repeat(np.fft.fft(Channels), self.RegenChannel, axis=0)
        # ChannelOutputs = np.concatenate(ChannelOutputs, axis=0)
        # return ChannelOutputs, Channels

if __name__ == '__main__':

    ChannelConfig = RayleighFadingConfig(
        velocity = 15,
        carrierFrequency = 2.5e9,
        delays_ns = [0, 100, 300],
        delayPower_dB = [0, -3, -6],
        Ts = 3.2e-8,
        T_slot = 1 / 30e-9,
        nOFDMSymbolsPerSlot = 14
    )

    TxWave = [np.arange(100)] * 14
    T_slot = 2e-3

    Channel = RayleighFadingChannel(ChannelConfig)
    ChannelOutputs = Channel.process(TxWave)



    
    # from dataclasses import dataclass
    
    # @dataclass
    # class AWGNChannelConfig():
    #     SNR: int = -6
    #     NumMC: int = 100000
    #     SeqLength: int = 10

    # class TestAWGNChannel():
    #     def __init__(self, config) -> None:
    #         self.NumMC = config.NumMC
    #         self.SeqLength = config.SeqLength
    #         self.SNR = config.SNR
    #         self.NoiseMixer = NoiseMixer()

    #     def run(self):
    #         symbols = np.random.randn(self.NumMC, self.SeqLength) + np.random.randn(self.NumMC, self.SeqLength) * 1j
    #         NoisySymbols= self.NoiseMixer.process(symbols, self.SNR)
    #         NoisePower = np.mean(np.abs(self.NoiseMixer.Noise) ** 2, axis=0)
    #         SymbolPower = np.mean(np.abs(symbols) ** 2, axis=0)
    #         EstimatedSNR = 10 * np.log10(SymbolPower / NoisePower)
    #         SNRError = EstimatedSNR - self.SNR
    #         return SNRError

    # config1 = AWGNChannelConfig()
    # SNRError = TestAWGNChannel(config1).run()
    # print("SNR Error: {}".format(SNRError))
