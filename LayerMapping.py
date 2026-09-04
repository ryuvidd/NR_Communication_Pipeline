import numpy as np

class LayerMapper():
    def __init__(self, nLayer:int):
        if not 1 <= nLayer <= 8:
            raise ValueError("Number of layers must be between 1 and 8.")
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
        # follows TS38.211 clause 7.3.1.3
        if len(QAMSymbols) > 2:
            raise ValueError("Number of codewords exceeds two.")
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
        
class LayerDemapper():

    def __init__(self, nLayer: int):
        if not 1 <= nLayer <= 8:
            raise ValueError("Number of layers must be between 1 and 8.")
        self.nLayer = nLayer
        layer_split = {
            5: (2, 3),
            6: (3, 3),
            7: (3, 4),
            8: (4, 4)
        }
        if nLayer >= 5:
            self.d0_num, self.d1_num = (layer_split[nLayer])

    @staticmethod
    def __combine_layers__(layers: list[np.ndarray]) -> np.ndarray:

        nLayers = len(layers)
        lengths = [len(layer) for layer in layers]
        if len(set(lengths)) != 1:
            raise ValueError("All layers belonging to the same codeword must have the same length.")

        symbols_per_layer = lengths[0]
        output = np.empty(nLayers * symbols_per_layer, dtype=layers[0].dtype)

        for layer_idx, layer in enumerate(layers):
            output[layer_idx::nLayers] = layer

        return output

    def process(self, LayerMappedSymbols: list[np.ndarray]) -> list[np.ndarray]:

        if len(LayerMappedSymbols) != self.nLayer:
            raise ValueError(f"Expected {self.nLayer} layers, but received {len(LayerMappedSymbols)}.")

        if self.nLayer <= 4:
            d0 = self.__combine_layers__(LayerMappedSymbols)
            return [d0]
        else:
            d0_layers = LayerMappedSymbols[:self.d0_num]
            d1_layers = LayerMappedSymbols[self.d0_num:]

            d0 = self.__combine_layers__(d0_layers)
            d1 = self.__combine_layers__(d1_layers)

            return [d0, d1]
    
if __name__ == '__main__':
    nLayer = 6
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
    LayerMappedSymbols = ThisLayerMapper.process(QAMSymbols)
    for i in range(nLayer):
        print(LayerMappedSymbols[i])

    print("\n--- Estimated QAM ---")
    ThisLayerDemapper = LayerDemapper(nLayer)
    EstimatedQAMSymbols = ThisLayerDemapper.process(LayerMappedSymbols)
    for i in range(q):
        print(f"Codeword {i}:")
        print("Maximum absolute difference: ", np.max(np.abs(QAMSymbols[i] - EstimatedQAMSymbols[i])))
        print("No significant error: ", np.allclose(QAMSymbols[i], EstimatedQAMSymbols[i])) 
        print("")
    