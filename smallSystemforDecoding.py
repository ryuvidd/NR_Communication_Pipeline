from LDPCEncoder import *
from LDPC_CBS import *

class transmitter():
    def __init__(self, TBS, R):
        self.meta = {}
        self.TBS = TBS
        self.R = R
        self.baseGraph = selectLDPCBaseGraph(TBS, R)
        self.CodeBlockSegmenter = CodeBlockSegmenter(self.baseGraph)
        self.meta["LDPCBlockParam"] = self.CodeBlockSegmenter.getLDPCBlockParam(TBS)
        temp_config = LDPCConfig(
            baseGraph = self.baseGraph,
            i_LS = self.meta["LDPCBlockParam"]["i_LS"],
            Z_c = self.meta["LDPCBlockParam"]["Z_c"],
            K = self.meta["LDPCBlockParam"]["K"],
            K_b = self.meta["LDPCBlockParam"]["K_b"]
        )
        self.Encoder = LDPCEncoder(temp_config)

    def process(self):
        rng = np.random.default_rng()
        TransportBlock = np.array(rng.integers(0, 2, size=self.TBS, dtype=np.int8))
        CodeBlocks = self.CodeBlockSegmenter.generateCodeBlocks(TransportBlock, self.meta["LDPCBlockParam"])
        EncodedCodeBlocks = self.Encoder.encode(CodeBlocks, validityCheckFlag=False)
        print('yeah')


if __name__ == "__main__":
    TBS = 100   
    R = 0.5

    Tx = transmitter(TBS, R)
    Tx.process()