import math, numpy as np, scipy
import matplotlib.pyplot as plt
import time


def multivariate_hypergeometric(N, K, n):
    """
    Simulate a sample from the multivariate hypergeometric distribution.
    
    Parameters:
    N (int): Total number of items.
    K (list of int): Number of items in each category (votes for each candidate).
    n (int): Number of items to sample.
    
    Returns:
    list of int: Sampled items from each category.
    """
    K = np.array(K)
    sampled = np.zeros_like(K)
    remaining_population = N
    remaining_sample = n

    for i in range(len(K) - 1): # last variable is degenerate
        sampled[i] = np.random.hypergeometric(K[i], remaining_population - K[i], remaining_sample)
        remaining_population -= K[i]  # equivalent to N1-n1
        remaining_sample -= sampled[i] # equivalent to K1-x1

        if remaining_sample == 0:
            for j in range(i+1,len(K)-1):
                sampled[j] = 0
            break

    sampled[-1] = remaining_sample
    return sampled


def doubly_multivariate_hypergeometric_simulate(N,K,n):
    """
    Simulate a sample from the doubly multivariate hypergeometric distribution.
    
    Parameters:
    N (int): Total number of items (votes).
    K (list of int): Number of items in each category (votes for each candidate).
    n (list of int): Number of items to sample in each sampling round (number of SA1s).
    
    Returns:
    array of int: Sampled items from each category for each sampling round.
    """

    X = np.empty((len(n), len(K)))

    # X_1 vector
    s = len(n)
    N_remaining = N
    K_remaining = K

    for i in range(s - 1): # 0,1,...,s-2 (altogether s-1):
        X_i = multivariate_hypergeometric(N_remaining, K_remaining, n[i])
        X[i] = X_i # add to X array

        N_remaining -= n[i]
        K_remaining = [a - b for a, b in zip(K_remaining, X_i)]


    X[s-1] = K_remaining

    return X

print(doubly_multivariate_hypergeometric_simulate(1000,[12,18,20,400,300,150,100],[5,88,307,500,100]))


# need to not only be able to simulate it, but to keep track of the function! Problem for tomorrow:)