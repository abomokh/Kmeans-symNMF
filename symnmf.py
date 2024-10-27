import sys
import numpy as np
import pandas as pd
import mysymnmfsp as s  # C module

# global arguments
goals = ['symnmf', 'sym', 'ddg', 'norm']

# initializing the random function
np.random.seed(0)


def print_matrix(matrix):
    """
    Prints a matrix to the console with elements rounded to 4 decimal places.

    Parameters:
    matrix (list of list of float): The matrix to be printed.
    """
    
    cols = len(matrix[0])
    for row in matrix:
        for i in range(cols-1):
            print(f"{np.round(row[i], 4):.4f}", end="")
            print(",", end="")
        print(f"{np.round(row[cols-1], 4):.4f}", end="")
        print()
    print()


def main():
    """
    Main function to execute the desired goal based on command line arguments.

    The function reads the goal, number of clusters (K), and input file from
    command line arguments. It then performs the specified matrix operation
    (symnmf, sym, ddg, or norm) and prints the resulting matrix.
    """

    inputs = sys.argv
 
    ## assertions ##
    if(len(sys.argv) != 4):
        print("An Error Has Occurred")
        return

    try:
        K = int(inputs[1])
    except ValueError:
        print("An Error Has Occurred")
        return
    
    try:
        goal = inputs[2]
    except ValueError:
        print("An Error Has Occurred")
        return
    
    if goal not in goals:
        print("An Error Has Occurred")
        return

    try:
        file = inputs[3]
    except ValueError:
        print("An Error Has Occurred")
        return
    
    if not file.endswith(".txt"):
        print("An Error Has Occurred")
        return
    
    datapoints = pd.read_csv(file, header=None).values.tolist()
    N = len(datapoints)

    if goal == "symnmf":
        W = np.array(s.norm(datapoints))
        m = np.mean(W)
        H = [[0 for i in range(K)] for j in range(N)]
        for i in range(N):
            for j in range(K):
                H[i][j] = np.random.uniform(0, 2*np.sqrt(m/K))
        symnmf_matrix = s.symnmf(H, W.tolist(), N, K)
        print_matrix(symnmf_matrix)

    elif goal == "sym":
        sym_matrix = s.sym(datapoints)
        print_matrix(sym_matrix)

    elif goal == "ddg":
        diagonal_matrix = s.ddg(datapoints)
        print_matrix(diagonal_matrix)

    else:
        norm_matrix = s.norm(datapoints)
        print_matrix(norm_matrix)


if __name__ == "__main__":
    main()