from dataclasses import dataclass
import numpy as np
from BG_loader import *
from collections import Counter
import logging

"""
===== LDPC Coding =====
following TS38.212 clause 5.3.2

- Solve w where H*x = 0 and x = concat([c, w]). Addition here is XOR operator in GF(2).
- c is the code block to be encoded, treating NULL as 0 for the rest of encoding process.
- A matrix H_BG has shape of 46x68 (for BG1) or 42x52 (for BG2) obtained from the table provided in 5.3.2-2 and 5.3.2-3, loaded using BG_loader (they already did the mod Z_c). 
    - For BG1, H_BG.shape = (46, 68), The num_col = 68 means num_info_block + num_parity, where num_info_block = 22 and num_parity = 46.
    - Each entry in H_BG tells how many circulant right-shifts performed on corresponding entry x. 
        - In practical, c.shape = (K_b, Z_c) not K_b * Z_c as in equation as the operation performs as group of Z_c.
        - and w.shape = (num_parity, Z_c)
        - e.g. H_BG[0,0] * c[0] outputs the Z_c-sized vector equals to c[0] that is circular right-shifted bt H_BG[0,0] times.
- H*x = 0 -> H_c*c + H_p*w = 0 -> (in GF(2)) H_p*w = H_c*c, where H_c = H_BG[:,:K_b] and H_p = H_BG[:,K_b:]
- First compute RHS = H_c*c whose shape is (K_b, Z_c)
    - by math, RHS[i] = (H_c[i,0] * c[0]) XOR (H_c[i,1] * c[1]) XOR ... XOR (H_c[i,K_b] * c[K_b])
- Then compute the parities w_blocks whose shape is (num_parity, Z_c)
    - compute w[0] first by using the first 4 rows of H_p
    - find the rest w[1]...w[num_parity]
- Full code block is concat([c.reshape(-1), w.reshape(-1)]) and apply the NULL back to the same indices
- Final encoded code block is fullCodeBlock[2*Z_c:].
- Checking validity follows H*x = 0 and x = concat([c, w]) where x is the full code block NOT encoded code block.
        
"""

@dataclass
class LDPCConfig:
    baseGraph: int
    i_LS: int 
    Z_c: int
    K: int

class LDPCEncoder():
    def __init__(self, config: LDPCConfig):
        self.baseGraph = config.baseGraph
        if self.baseGraph not in (1,2):
            raise ValueError("Base graph must be 1 or 2.")
        self.i_LS = config.i_LS
        self.Z_c = config.Z_c
        self.K = config.K
        self.num_info_block = 22 if self.baseGraph == 1 else 10
        self.H_BG = loadLDPCBaseGraph(config.baseGraph, config.i_LS, config.Z_c)
    
    def __compute_RHS__(self, H_c: np.ndarray, c_blocks: np.ndarray) -> np.ndarray:
        M, K_b = H_c.shape
        RHS = np.zeros((M, self.Z_c), dtype=np.uint8)
        for i in range(M):
            for j in range(K_b):
                shift = H_c[i,j]
                if shift != -1:
                    RHS[i] ^= np.roll(c_blocks[j], shift)
        return RHS
    
    def __getShift_w0__(self, H_p: np.ndarray):
        shifts = H_p[:4, 0]
        shifts = [s for s in shifts if s != -1]
        counts = Counter(shifts)
        remaining = [shift for shift, count in counts.items() if count % 2 ==1]
        if len(remaining) == 1: 
            return remaining[0]
        else: raise ValueError("Could not isolate p0 to exactly one circulant permutation.")
    
    def __compute_w0__(self, H_p: np.ndarray, RHS: np.ndarray):
        combined_RHS = (RHS[0] ^ RHS[1] ^ RHS[2] ^ RHS[3])
        shift = self.__getShift_w0__(H_p)
        w0 = np.roll(combined_RHS, -shift)
        return w0
    
    def __find_single_unknown_equation__(self, H_p: np.ndarray, known_w_blocks: np.ndarray):
        num_row, num_parity = H_p.shape
        for row in range(num_row):
            unknown = []
            for p_idx in range(num_parity):
                shift = H_p[row, p_idx]
                if shift != -1 and not known_w_blocks[p_idx]:
                    unknown.append(p_idx)
            if len(unknown) == 1:
                return row, unknown[0]
        return None, None
    
    def __solve_single_unknown__(self, H_p:np.ndarray, RHS:np.ndarray, w_blocks:np.ndarray, known_w_blocks: np.ndarray, row: int, target: int):

        value = RHS[row].copy()
        target_shift = H_p[row, target]
        num_parity = H_p.shape[1]

        if target_shift == -1: raise ValueError("Target block is not connected to this row.")

        for p_idx in range(num_parity):
            if p_idx == target: continue
            if not known_w_blocks[p_idx]: continue
            shift = H_p[row, p_idx]
            if shift != -1: 
                value ^= np.roll(w_blocks[p_idx], shift)

        w_target = np.roll(value, -target_shift)
        return w_target
    
    def __parity_check__(self, fullCodeBlock: np.ndarray) -> bool:
        N = 66 * self.Z_c if self.baseGraph == 1 else 50 * self.Z_c
        if (len(fullCodeBlock) - 2*self.Z_c) != N : 
            raise ValueError("Incorrect output length")

        num_row, num_col = self.H_BG.shape
        x_blocks = fullCodeBlock.copy()
        x_blocks[x_blocks == -1] = 0
        x_blocks = x_blocks.astype(np.uint8).reshape(num_col, self.Z_c)
        syndrome = np.zeros((num_row, self.Z_c), dtype=np.uint8)
        for i in range(num_row):
            for j in range(num_col):
                shift = self.H_BG[i,j]
                if shift != -1:
                    syndrome[i] ^= np.roll(x_blocks[j], shift)
        if np.all(syndrome == 0): 
            print("Encoded code block is valid.")
            return True
        else: return False
    
    def process(self, codeBlocks: list, validityCheckFlag: bool) -> list:
        ## Following TS38.212 clause 5.3.2: LDPC encoding ##
        encodedCodeBlocks = []
        for codeBlock in codeBlocks:
            if len(codeBlock) != self.K: raise ValueError(f"Code block length must be {self.K}.")

            # Treat NULL as 0 first
            NewcodeBlock = codeBlock.copy()
            fillerMask = (NewcodeBlock == -1)
            NewcodeBlock[fillerMask] = 0
            NewcodeBlock = NewcodeBlock.astype(np.uint8)

            H_c = self.H_BG[:,:self.num_info_block]
            H_p = self.H_BG[:,self.num_info_block:]
            
            # Compute Parities
            c_blocks = NewcodeBlock.reshape(-1, self.Z_c)
            RHS = self.__compute_RHS__(H_c, c_blocks)

            num_parity = H_p.shape[1]
            w_blocks = np.zeros((num_parity, self.Z_c), dtype=np.uint8)
            known_w_blocks = np.zeros(num_parity, dtype=bool)

            w_blocks[0] = self.__compute_w0__(H_p, RHS)
            known_w_blocks[0] = True

            while not np.all(known_w_blocks):
                row, target = self.__find_single_unknown_equation__(H_p, known_w_blocks)
                if row is None or target is None:
                    raise RuntimeError("No equation with exactly one unknown parity block.")
                w_blocks[target] = self.__solve_single_unknown__(H_p, RHS, w_blocks, known_w_blocks, row, target)
                known_w_blocks[target] = True
            
            # Concatenate code block with parities and apply NULL, following step 2
            fullCodeBlock = np.concatenate([NewcodeBlock, w_blocks.reshape(-1)])
            fullCodeBlock = fullCodeBlock.astype(np.int8)
            fullCodeBlock[:self.K][fillerMask] = -1
            encodedCodeBlock = fullCodeBlock.copy()[2 * self.Z_c:]

            if validityCheckFlag:
                logging.info("..... Checking validity of encoded code block .....")
                isValid = self.__parity_check__(fullCodeBlock)
                if isValid: 
                    logging.info("===== Encoded code block is valid ======")
                    encodedCodeBlocks.append(encodedCodeBlock)
                else: 
                    raise ValueError("Encoded code block is invalid")
            else:
                encodedCodeBlocks.append(encodedCodeBlock)
        return encodedCodeBlocks
    
class LDPCDecoder():
    def __init__(self, config: LDPCConfig, maxIter: int, fillerMask=None):
        self.baseGraph = config.baseGraph
        if self.baseGraph not in (1,2):
            raise ValueError("Base graph must be 1 or 2.")
        self.i_LS = config.i_LS
        self.Z_c = config.Z_c
        self.K = config.K
        self.num_info_block = 22 if self.baseGraph == 1 else 10
        self.H_BG = loadLDPCBaseGraph(config.baseGraph, config.i_LS, config.Z_c)
        self.maxIter = maxIter
        self.fillerMask = fillerMask

    def __solve_min_sum__(self, H, L, row_messages):
        """
        Layered min-sum update for one parity-check row.

        H : shape (num_variables,)
            Shift values for this check row.
        L : shape (num_variables, Zc)
            Current total LLRs.
        row_messages : shape (num_variables, Zc)
            Stored check-to-variable messages for this check row.
        """

        connected = np.where(H != -1)[0]
        for i in connected:
            # Remove old CN -> VN message
            q = L[i] - row_messages[i]
            messages = []
            for j in connected:
                if j == i:
                    continue
                # Remove old message from this edge
                q_j = L[j] - row_messages[j]
                # Align all variable messages to check-node domain
                messages.append(np.roll(q_j, H[j]))
            messages = np.asarray(messages)

            # Min-sum
            sign = np.prod(np.sign(messages), axis=0)
            magnitude = np.min(np.abs(messages), axis=0)
            r_new = sign * magnitude

            # Convert back to VN orientation
            r_new = np.roll(r_new, -H[i])

            # Update total LLR
            L[i] = q + r_new

            # Store CN -> VN message
            row_messages[i] = r_new

        return L, row_messages
    
    def __parity_check__(self, x_hard) -> bool:
        num_row, num_col = self.H_BG.shape
        x_blocks = x_hard.copy()
        x_blocks[x_blocks == -1] = 0
        x_blocks = x_blocks.astype(np.uint8).reshape(num_col, self.Z_c)
        syndrome = np.zeros((num_row, self.Z_c), dtype=np.uint8)
        for i in range(num_row):
            for j in range(num_col):
                shift = self.H_BG[i,j]
                if shift != -1:
                    syndrome[i] ^= np.roll(x_blocks[j], shift)
        if np.all(syndrome == 0): return True
        else: return False

    def process(self, EncodedLLRs: list[np.ndarray]):
        if self.fillerMask is not None and len(EncodedLLRs) != len(self.fillerMask):
            raise ValueError("Number of LLR blocks does not match number of filler masks.")

        EstimatedCodewords = []
        for cb_index, encoded_llrs in enumerate(EncodedLLRs):
            L = np.concatenate((np.zeros(2 * self.Z_c), encoded_llrs)).reshape(-1, self.Z_c)
            R = np.zeros((self.H_BG.shape[0], self.H_BG.shape[1], self.Z_c), dtype=np.float64)
            converged = False

            for iter_count in range(self.maxIter):
                for row in range(self.H_BG.shape[0]):
                    L, R[row] = self.__solve_min_sum__(self.H_BG[row], L, R[row])
                x_hard = (L < 0).astype(np.uint8)

                if self.__parity_check__(x_hard):
                    converged = True
                    break

            if not converged:
                print(f"Block {cb_index}: decoder did not converge after {self.maxIter} iterations")

            estimated_codeword = x_hard[:self.num_info_block].reshape(-1).astype(np.int8)
            if self.fillerMask is not None:
                mask = self.fillerMask[cb_index]
                if len(estimated_codeword) != len(mask):
                    raise ValueError(f"Filler mask length does not match decoded block {cb_index}.")
                estimated_codeword[mask] = -1
            EstimatedCodewords.append(estimated_codeword)
        
        return EstimatedCodewords

        
if __name__ == "__main__":
    baseGraph = 1
    i_LS = 7
    Z_c = 60
    num_testing_cb = 10
    rng = np.random.default_rng(23)

    config = LDPCConfig(
        baseGraph = baseGraph,
        i_LS = i_LS,
        Z_c = Z_c,
        K = 22 * Z_c if baseGraph == 1 else 10 * Z_c
    )

    num_null = int(config.K*0.1)
    print(f"Number of testing code blocks: {num_testing_cb}")
    print(f"Number of NULL: {num_null}")
    codeBlocks = []
    for r in range(num_testing_cb):
        codeBlock = rng.integers(0, 2, size=config.K, dtype=np.int8)
        codeBlock[-num_null:] = -1
        codeBlocks.append(codeBlock)
    
    Encoder = LDPCEncoder(config)
    EncodedBlocks = Encoder.process(codeBlocks, validityCheckFlag=True)

    EncodedLLRs = []
    for encoded_block in EncodedBlocks:
        encoded_llrs = np.zeros(len(encoded_block), dtype=np.float64)
        encoded_llrs[encoded_block == 0] = 2.0
        encoded_llrs[encoded_block == 1] = -2.0
        EncodedLLRs.append(encoded_llrs)

    Decoder = LDPCDecoder(config, 20, )
    EstimatedCodeWord = Decoder.process(EncodedLLRs)
    
    all_recovered = True
    for i in range(len(codeBlocks)):
        expected_codeword = codeBlocks[i].copy()
        expected_codeword[expected_codeword == -1] = 0
        recovered = np.array_equal(EstimatedCodeWord[i], expected_codeword)
        all_recovered &= recovered
        print(f"Block {i+1}: systematic bits recovered: {recovered}")

    print(f"All {num_testing_cb} code blocks recovered: {all_recovered}")
    
