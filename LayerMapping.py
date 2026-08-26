import numpy as np

class LayerMapper():
    def __init__(self, nLayer:int):
        self.nLayer = nLayer
        layer_split = {
            5: (2, 3),
            6: (3, 3),
            7: (3, 4),
            8: (4, 4)
        }
        if nLayer >= 5: 
            self.d0_num, self.d1_num = (layer_split[nLayer])

    def process(self, QAMSymbols: list[np.ndarray]) -> list[np.ndarray]:
        if len(QAMSymbols) > 2:
            raise ValueError("Number of QAM modulated codeword exceeds two.")
        if self.nLayer <= 4 and len(QAMSymbols) != 1:
            raise ValueError("1 to 4 layers require exactly one codeword.")
        if self.nLayer >= 5 and len(QAMSymbols) != 2:
            raise ValueError("5 to 8 layers require exactly two codewords.")
        
        if self.nLayer <= 4:
            d0 = QAMSymbols[0]
            x = [d0[layer_idx::self.nLayer] for layer_idx in range(self.nLayer)]
            return x
        else:
            d0 = QAMSymbols[0]
            d1 = QAMSymbols[1]
            x = [d0[layer_idx::self.d0_num] for layer_idx in range(self.d0_num)]
            x.extend([d1[layer_idx::self.d1_num] for layer_idx in range(self.d1_num)])
            return x
    
if __name__ == '__main__':
    nLayer = 8
    seqlen = nLayer * 10
    seq = np.arange(seqlen)
    q = 2 if nLayer > 4 else 1
    q_len = seqlen // q
    QAMSymbols = []
    print("--- QAMSymbols ---")
    for i in range(q):
        cw = seq[i*q_len:(i+1)*q_len]
        QAMSymbols.append(cw)
        print(cw)

    print("\n--- Layer Mapped ---")
    ThisLayerMapper = LayerMapper(nLayer)
    output = ThisLayerMapper.process(QAMSymbols)
    for i in range(nLayer):
        print(output[i])