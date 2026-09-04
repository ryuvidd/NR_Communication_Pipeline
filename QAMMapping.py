import numpy as np

class QAMMapper():
    def __init__(self, Qm):
        if Qm in (2,4,6,8): self.Qm = Qm
        else: raise ValueError("Qm is not 2,4,6, or 8.")

    def process(self, ScrambledBits: list[np.ndarray]) -> list[np.ndarray]:
        nCodeword = len(ScrambledBits)
        ModulatedSymbols = []
        for i in range(nCodeword):
            Bits = ScrambledBits[i].astype(np.int8)
            if self.Qm == 2:  # QPSK
                I = 1 - 2*Bits[::2]
                Q = 1 - 2*Bits[1::2]
                normalization = 2
            elif self.Qm == 4:  # 16QAM
                I = (1 - 2*Bits[::4]) * (2 - (1 - 2*Bits[2::4]))
                Q = (1 - 2*Bits[1::4]) * (2 - (1 - 2*Bits[3::4]))
                normalization = 10
            elif self.Qm == 6:  # 64QAM
                I = (1 - 2*Bits[::6]) * (4 - (1 - 2*Bits[2::6])*(2 - (1 - 2*Bits[4::6])))
                Q = (1 - 2*Bits[1::6]) * (4 - (1 - 2*Bits[3::6])*(2 - (1 - 2*Bits[5::6])))
                normalization = 42
            else:  # 256QAM
                I = (1 - 2*Bits[::8]) * (8 - (1 - 2*Bits[2::8])*(4 - (1 - 2*Bits[4::8])*(2 - (1 - 2*Bits[6::8]))))
                Q = (1 - 2*Bits[1::8]) * (8 - (1 - 2*Bits[3::8])*(4 - (1 - 2*Bits[5::8])*(2 - (1 - 2*Bits[7::8]))))
                normalization = 170
            symbols = (I + 1j*Q) / np.sqrt(normalization)
            ModulatedSymbols.append(symbols)
        return ModulatedSymbols
    
class QAMDemapper:
    def __init__(self, Qm: int):
        if Qm not in (2, 4, 6, 8):
            raise ValueError("Qm must be 2, 4, 6, or 8.")
        self.Qm = Qm
        self.constellation, self.bit_labels = self.__generate_constellation__()

    def __generate_constellation__(self):
        M = 2 ** self.Qm
        # Generate all possible bit combinations
        indices = np.arange(M)
        bit_labels = ((indices[:, None] >> np.arange(self.Qm - 1, -1, -1)) & 1).astype(np.uint8)

        b = bit_labels.astype(np.int8)
        if self.Qm == 2:
            I = 1 - 2*b[:,0]
            Q = 1 - 2*b[:,1]
            normalization = np.sqrt(2)
        elif self.Qm == 4:
            I = (1 - 2*b[:, 0]) * (2 - (1 - 2*b[:, 2]))
            Q = (1 - 2*b[:, 1]) * (2 - (1 - 2*b[:, 3]))
            normalization = np.sqrt(10)
        elif self.Qm == 6:
            I = (1 - 2*b[:, 0]) * (4 - (1 - 2*b[:, 2]) * (2 - (1 - 2*b[:, 4])))
            Q = (1 - 2*b[:, 1]) * (4 - (1 - 2*b[:, 3]) * (2 - (1 - 2*b[:, 5])))
            normalization = np.sqrt(42)
        else:  # Qm == 8
            I = (1 - 2*b[:, 0]) * (8 - (1 - 2 * b[:, 2]) * (4 - (1 - 2 * b[:, 4]) * (2 - (1 - 2 * b[:, 6]))))
            Q = (1 - 2*b[:, 1]) * (8 - (1 - 2 * b[:, 3]) * (4 - (1 - 2 * b[:, 5]) * (2 - (1 - 2 * b[:, 7]))))
            normalization = np.sqrt(170)

        constellation = (I + 1j * Q) / normalization
        return constellation, bit_labels

    def process(self, ReceivedSymbols: list[np.ndarray], noiseVariance: float) -> list[np.ndarray]:
        if noiseVariance <= 0:
            raise ValueError("noiseVariance must be positive.")

        codewordLLRs = []
        for symbols in ReceivedSymbols:
            # Shape:
            #
            # (N_symbols, M)
            #
            # Each row contains distances from one
            # received symbol to every constellation point.

            distances = np.abs(symbols[:, None] - self.constellation[None, :]) ** 2
            LLRs = np.empty((len(symbols), self.Qm), dtype=np.float64)
            for bit_position in range(self.Qm):
                mask_0 = (self.bit_labels[:, bit_position] == 0)
                mask_1 = (self.bit_labels[:, bit_position] == 1)

                min_distance_0 = np.min(distances[:, mask_0], axis=1)
                min_distance_1 = np.min(distances[:, mask_1], axis=1)

                # Max-log approximation
                #
                # LLR > 0 → bit 0 more likely
                # LLR < 0 → bit 1 more likely

                LLRs[:, bit_position] = (min_distance_1 - min_distance_0) / noiseVariance

            codewordLLRs.append(LLRs.reshape(-1))
        return codewordLLRs

if __name__ == '__main__':

    NumBits = 100000
    Qm = 2
    bits = np.random.randint(0, 2, NumBits)
    ThisQAMMapper = QAMMapper(Qm)
    ModulatedSymbols = ThisQAMMapper.process([bits])

    ThisQAMDemapper = QAMDemapper(Qm)
    LLRs = ThisQAMDemapper.process(ModulatedSymbols, noiseVariance=1e-10)
    print('Assume ideal mapping...')
    EstimatedBits = np.array([1 if llr < 0 else 0 for llr in LLRs[0]])
    print("Maximum absolute difference: ", np.max(np.abs(bits - EstimatedBits)))
    print("No significant error: ", np.allclose(bits, EstimatedBits))



