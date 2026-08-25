def selectLDPCBaseGraph(A: int, R: float) -> str:
    if A <= 292: return "BG2"
    elif R <= 0.25: return "BG2"
    elif A <= 3824 and R <= 0.67: return "BG2"
    else: return "BG1"

if __name__ == '__main__':
    A = 13064
    R = 0.5
    baseGraph = selectLDPCBaseGraph(A, R)
    print(baseGraph)