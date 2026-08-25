import numpy as np

class CRC:
    def __init__(self, L) -> None:
        self.Generator = self.selectGenerator(L)
        self.CRCLength = len(self.Generator) - 1
        pass

    def selectGenerator(self, L):
        if L == '6':  
            oneIndices = [0,5,6]
            lengthGenerator = 7
        elif L == '11': 
            oneIndices = [0,5,9,10,11]
            lengthGenerator = 12
        elif L == '16': 
            oneIndices = [0,5,12,16]
            lengthGenerator = 17
        elif L == '24A': 
            oneIndices = [0,1,3,4,5,6,7,10,11,14,17,18,23,24]
            lengthGenerator = 25
        elif L == '24B': 
            oneIndices = [0,1,5,6,23,24]
            lengthGenerator = 25
        elif L == '24C': 
            oneIndices = [0,1,2,4,8,12,13,15,17,20,21,23,24]
            lengthGenerator = 25
        else: raise(ValueError)
        return [1 if e in oneIndices else 0 for e in range(lengthGenerator)]
    
    def attachCRC(self, dataBits: np.ndarray) -> tuple:
        ## Following TS38.212 clause 5.2.2: LDPC CRC and CBS ##
        dividend = np.concatenate((dataBits, np.zeros(self.CRCLength, dtype=dataBits.dtype)))
        remainder = dividend.copy()
        for i in range(len(dataBits)):
            if remainder[i] == 1:
                for j in range(len(self.Generator)):
                    remainder[i+j] ^= self.Generator[j]
        crc = remainder[-self.CRCLength:]
        codeWord = np.concatenate((dataBits, crc))
        return codeWord, crc
    
    def check(self, codeWord):
        remainder = codeWord.copy()
        for i in range(len(codeWord)-self.CRCLength):
            if remainder[i] == 1:
                for j in range(len(self.Generator)):
                    remainder[i+j] ^= self.Generator[j]
        crc_remainder = remainder[-self.CRCLength:]
        return all(bit == 0 for bit in crc_remainder)
    
if __name__ == '__main__':

    data_bits = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    L = '6'

    # Encode
    ThisCRC = CRC(L)
    codeword, crc = ThisCRC.attachCRC(data_bits)

    print("Information bits :", data_bits)
    print("Generator        :", ThisCRC.Generator)
    print("CRC              :", crc)
    print("Codeword         :", codeword)

    # Check without error
    result = ThisCRC.check(codeword)
    print("\nCRC check (no error):", result)

    # Introduce a bit error
    received = codeword.copy()
    received[2] ^= 1
    print("\nReceived bits with error:", received)

    result = ThisCRC.check(received)
    print("CRC check (with error):", result)
