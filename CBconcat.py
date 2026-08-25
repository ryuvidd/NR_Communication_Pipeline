import numpy as np

class CodeBlockConcatenator():
    def process(self, rateMatchedCodeBlocks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(rateMatchedCodeBlocks)