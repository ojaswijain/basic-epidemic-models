"""
Code: BTech Project Thesis
Author: Ojaswi Jain
Guide: Prof. Ajit Rajwade
Date: 20th Aug 2023
"""

"""
Convex optimized solution for robust fitting


MODIFICATION TO KEEP BETA/GAMMA CONSTANT:
Convert the above problem to:
    minSum(|b(t)-b0|*S(t)*I(t) + |g(t)-k*b0|*I(t))
where k is a constant

Implementation of the above in piecewise form
Some points are missing in the data, so we need to interpolate them
Can interpolate expected values of S, I, R & beta, gamma; or interpolate calculated dS, dI, dR
Implementing the former
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from scipy.optimize import minimize
import argparse

# Defining constants
N = 2e7
RECOVERY = 14
epsilon = 1e-7
limit = 0
train = 0.8
test = 0.2
window = 7
weeks = 42
miss = True
frac_miss = 0.1

#set seed
np.random.seed(0)

def calc_deltas(S, I, R):
    # S[i] = S[i-1] + dS[i]
    dS = np.zeros(len(S))
    dI = np.zeros(len(I))
    dR = np.zeros(len(R))
    dS[1:] = S[1:] - S[:-1]
    dI[1:] = I[1:] - I[:-1]
    dR[1:] = R[1:] - R[:-1]
    return dS, dI, dR

def plot_true_beta_gamma(beta, gamma, S, I, R, S_cum, I_cum, R_cum, train, idx):
    """
    @params: beta, gamma, R0
    @returns: None
    """
    #Plot True values of dS, and -beta*S*I
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='true dS/dt')
    if miss == True:
        vals = -beta[idx-1]*S_cum[idx-1]*I_cum[idx-1]
        plt.scatter(idx, vals, color='r', label='Missing Values')
    plt.plot(pd.Series(S).rolling(window=7).mean(), label='Weekly rolling mean of dS/dt')
    plt.xlabel('Days')
    plt.ylabel('dS/dt')
    plt.title('True dS/dt vs -beta*S*I')
    plt.legend()
    plt.show()

    #Plot True values of dI, and beta*S*I - gamma*I
    plt.figure(figsize=(10, 6))
    plt.plot(I, label='true dI/dt')
    if miss == True:
        vals = beta[idx-1]*S_cum[idx-1]*I_cum[idx-1] - gamma[idx-1]*I_cum[idx-1]
        plt.scatter(idx, vals, color='r', label='Missing Values')
    plt.plot(pd.Series(I).rolling(window=7).mean(), label='Weekly rolling mean of dI/dt')
    plt.xlabel('Days')
    plt.ylabel('dI/dt')
    plt.legend()
    plt.title('True dI/dt vs beta*S*I - gamma*I')
    plt.show()

    #Plot True values of dR, and gamma*I
    plt.figure(figsize=(10, 6))
    plt.plot(R, label='true dR/dt')
    if miss == True:
        vals = gamma[idx-1]*I_cum[idx-1]
        plt.scatter(idx, vals, color='r', label='Missing Values')
    plt.plot(pd.Series(R).rolling(window=7).mean(), label='Weekly rolling mean of dR/dt')
    plt.xlabel('Days')
    plt.ylabel('dR/dt')
    plt.legend()
    plt.title('True dR/dt vs gamma*I')
    plt.show()

def interpolate_SIR(S_cum, I_cum, R_cum, beta, gamma, idx):
    """
    @params: S_cum, I_cum, R_cum
    @returns: S_cum, I_cum, R_cum

    Use the calculated S,I,R, beta, gamma of previous day to interpolate missing values
    """

    #interpolate missing data

    S_cum[idx] = S_cum[idx-1] - beta[idx-1]*S_cum[idx-1]*I_cum[idx-1]
    I_cum[idx] = I_cum[idx-1] + beta[idx-1]*S_cum[idx-1]*I_cum[idx-1] - gamma[idx-1]*I_cum[idx-1]
    R_cum[idx] = R_cum[idx-1] + gamma[idx-1]*I_cum[idx-1]

    return S_cum, I_cum, R_cum

def interpolate_beta_gamma(S, R, S_cum, I_cum, R_cum, beta, gamma, idx):
    """
    @params: beta, gamma
    @returns: beta, gamma
    """
    beta[idx] = S[idx+1]/(-S_cum[idx]*I_cum[idx])
    gamma[idx] = R[idx+1]/I_cum[idx]

    return beta, gamma

def calc_beta(dS, S, I):
    """
    @params: dS/dt, S, I
    @returns: beta based on best fit
    """
    beta = np.zeros(len(dS))
    beta[:-1] = -dS[1:]/(I[:-1]*S[:-1])
    #if S or I = 0 then beta = 0
    beta = np.nan_to_num(beta, copy=True, posinf=limit, neginf=-limit)
    # Replace with piecewise median, piece size = window while keeping array length same
    num_pieces = len(beta) // window
    beta_pieces = np.array_split(beta[:num_pieces * window], num_pieces)
    # Calculate the median for each piece and create a new array repeating medians
    beta = np.concatenate([np.full(window, np.median(piece)) for piece in beta_pieces])
    return beta

def calc_gamma(dR, I):
    """
    @params: dR, I
    @returns: gamma based on best fit
    """
    gamma = np.zeros(len(dR))
    gamma[:-1] = dR[1:]/I[:-1]
    gamma = np.nan_to_num(gamma, copy=True, posinf=limit, neginf=-limit)
    #Replace with piecewise median, piece size = window while keeping array length same
    num_pieces = len(gamma) // window
    gamma_pieces = np.array_split(gamma[:num_pieces * window], num_pieces)
    # Calculate the median for each piece and create a new array repeating medians
    gamma = np.concatenate([np.full(window, np.median(piece)) for piece in gamma_pieces])
    return gamma


def calc_R0(beta, gamma):
    """
    @params: beta, gamma
    @returns: R0
    """
    R0 = beta/gamma
    R0 = np.nan_to_num(R0, copy=True, posinf=limit, neginf=-limit)
    return R0

def plot_beta_gamma(beta, gamma, R0):
    """
    @params: beta, gamma, R0
    @returns: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(beta, label='Beta')
    plt.plot(gamma, label='Gamma')
    # plt.plot(R0, label='R0')
    #plot medians of beta and gamma
    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.title('SIR Parameters: Beta, Gamma, R0')
    plt.legend()
    plt.show()

def plot_SIR(S_cum, I_cum, R_cum):
    """
    @params: S, I, R
    @returns: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(S_cum, label='S')
    plt.plot(I_cum, label='I')
    plt.plot(R_cum, label='R')
    plt.xlabel('Days')
    plt.ylabel('Fraction of Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()

def read_data(file):
    """
    @params: None
    @returns: S, I, R, S_cum, I_cum, R_cum
    """
    # Reading data
    df = pd.read_csv(file)
    # days = weeks*window
    # df = df[:days]
    R_cum = (df["Cured"].to_numpy()+df["Deaths"].to_numpy())/N
    I_cum = df["Confirmed"].to_numpy()/N - R_cum
    S_cum = 1-I_cum-R_cum
    S, I, R = calc_deltas(S_cum, I_cum, R_cum)
    return S, I, R, S_cum, I_cum, R_cum

def main(file):
    """
    @params: None
    @returns: None
    """
    # Reading data
    S, I, R, S_cum, I_cum, R_cum = read_data(file)

    #idx of missing values (20 random values)
    idx = np.random.randint(50, 250, 20)

    weights = np.copy(I_cum)
    weights[idx] = 0

    # Calculating beta and gamma
    beta = calc_beta(S, S_cum, weights)
    gamma = calc_gamma(R, weights)

    # S_cum, I_cum, R_cum = interpolate_SIR(S_cum, I_cum, R_cum, beta, gamma, idx)
    # beta, gamma = interpolate_beta_gamma(S, R, S_cum, I_cum, R_cum, beta, gamma, idx)
    miss_vals_I = beta[idx-1]*S_cum[idx-1]*I_cum[idx-1] - gamma[idx-1]*I_cum[idx-1]
    # mean_I = pd.Series(I).rolling(window=7).mean()
    # mean_err = np.abs(mean_I[idx] - miss_vals_I)/np.abs(mean_I[idx])
    # print("Error from mean", sorted(mean_err))
    # print("50th percentile error from the mean: ", np.percentile(mean_err, 50))
    err = np.abs(I[idx] - miss_vals_I)/np.abs(I[idx])
    #print all quartiles of error
    print(sorted(err))
    print("50th percentile error: ", np.percentile(err, 50))
    # print(idx-1)
    # print(I[idx])
    # print(miss_vals_I)
    # plt.hist(miss_vals_I/I[idx])
    # plt.show()
    # Plotting
    # plot_SIR(S_cum, I_cum, R_cum)
    plot_true_beta_gamma(beta, gamma, S, I, R, S_cum, I_cum, R_cum, 1, idx)
    # plot_beta_gamma(beta, gamma, calc_R0(beta, gamma))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../Data/delhi_2020.csv')
    parser.add_argument('--pop', type=int, default=1e7)
    args = parser.parse_args()
    file = args.file
    N = args.pop
    main(file)
