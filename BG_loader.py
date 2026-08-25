import numpy as np

def loadLDPCBaseGraph(baseGraph, i_LS, Z_c):
    file_path = "tables/NR_" + str(baseGraph) + "_" + str(i_LS) + "_" + str(Z_c) + ".txt"
    bg_matrix = np.full((46, 68), -1, dtype=int) if baseGraph == 1 else np.full((42, 52), -1, dtype=int)
    with open(file_path, 'r') as f:
        r = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            tokens = line.split()
            bg_matrix[r,:] = [int(e) for e in tokens]
            r += 1
                    
    return bg_matrix

if __name__ == "__main__":
    baseGraph = 1
    i_LS = 2
    Z_c = 320

    bg_matrix = loadLDPCBaseGraph(baseGraph, i_LS, Z_c)

    print("Matrix Shape:", bg_matrix.shape)
    print("First 5x5 block:\n", bg_matrix[:5, :5])