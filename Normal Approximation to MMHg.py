import math, numpy as np, scipy
import matplotlib.pyplot as plt
import time

N = 1000
K = [12,18,20,400,300,150,100]
n = [5,88,307,500,100]


mean_X = np.empty((len(n), len(K)))

for i in range(len(n)):
    for l in range(len(K)):
        mean_X[i][l] = n[i]*K[l]/N


# Flatten the array
mean_X_flat = mean_X.flatten()
print("Flattened array:", mean_X_flat, sum(mean_X_flat))


import numpy as np

def generate_covariance_matrix(N, K, n):
    """
    Generates a covariance matrix with specified structures.
    
    Parameters:
    - N
    - K
    - n
    
    Returns:
    - cov_matrix (ndarray): Covariance matrix of size (n_rows * n_cols) x (n_rows * n_cols).
    """

    n_rows = len(n) - 1
    n_cols = len(K) - 1
    size = n_rows * n_cols
    cov_matrix = np.zeros((size, size))
    
    for i in range(n_rows):
        for l in range(n_cols):
            for j in range(n_rows):
                for m in range(n_cols):
                    idx_1 = i * n_cols + l
                    idx_2 = j * n_cols + m
                    
                    if i == j and l == m:
                        cov_matrix[idx_1, idx_2] = n[i]*(N - n[i])*K[l]*(N - K[l])/(N^2*(N-1)) # var
                    elif i == j and l != m:
                        cov_matrix[idx_1, idx_2] = -n[i]*(N - n[i])*K[l]*K[m]/(N^2*(N-1))
                    elif i != j and l == m:
                        cov_matrix[idx_1, idx_2] = -n[i]*n[j]*K[l]*(N - K[l])/(N^2*(N-1))
                    elif i != j and l != m:
                        cov_matrix[idx_1, idx_2] = n[i]*n[j]*K[l]*K[m]/(N^2*(N-1)) # all different
                        
    return cov_matrix



# Generate covariance matrix
cov_matrix = generate_covariance_matrix(N,K,n)
print("Covariance matrix:\n", cov_matrix)
