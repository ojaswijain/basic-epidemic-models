"""
Code: BTech Project Thesis
Author: Ojaswi Jain
Guide: Prof. Ajit Rajwade
Date: 20th Aug 2023
"""

"""
Basic implementation of SIR epidemiological model, With beta/gamma as constant
Fundamental Constraints:
1. Total population is constant (S+I+R = N)
2. No Non-Pharmaceutical Interventions (NPIs)

Differential Equations:
dS/dt = -beta*S*I
dI/dt = beta*S*I - gamma*I
dR/dt = gamma*I

Available Data:
1. Total Population (N)
2. Newly Infected each day (delta_I)
3. (NEW) Newly Recovered/Dead each day (delta_R)

Parameters to be estimated:
1. Beta
2. Gamma
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Defining constants
N = 1e7
RECOVERY = 14
limit = 0
train = 0.8
test = 0.2
window = 7

# def calc_deltas(S, I, R):
#     """
#     @params: S: Susceptible population on each day
#              I: Infected population on each day
#              R: Recovered population on each day
#              S_cum: Cumulative Susceptible population on each day
#              I_cum: Cumulative Infected population on each day
#              R_cum: Cumulative Recovered population on each day
#     @returns: None
#     """
#     dS = np.zeros(len(S))
#     dI = np.zeros(len(I))
#     dR = np.zeros(len(R))
#     dI[1:] = I[1:] - I[:-1]
#     dS[1:] = -dI[1:]
#     dR[1:] = R[1:] - R[:-1]
#     dI -= dR
#     return dS, dI, dR

def calc_deltas(S, I, R):
    dS = np.zeros(len(S))
    dI = np.zeros(len(I))
    dR = np.zeros(len(R))
    dS[1:] = S[1:] - S[:-1]
    dI[1:] = I[1:] - I[:-1]
    dR[1:] = R[1:] - R[:-1]
    return dS, dI, dR

def plot(S, I, R, S_cum, I_cum, R_cum):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='Susceptible')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.plot(S+I+R, label='Total')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('SIR Model: New Cases')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(S_cum, label='Susceptible')
    plt.plot(I_cum, label='Infected')
    plt.plot(R_cum, label='Recovered')
    plt.plot(S_cum+I_cum+R_cum, label='Total')
    plt.xlabel('Days')
    plt.ylabel('Cumulative Population')
    plt.title('SIR Model: Cumulative Cases')
    plt.legend()
    plt.show()

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
    return R0

def plot_true_beta_gamma(beta, gamma, S, I, R, S_cum, I_cum, R_cum, train):
    """
    @params: beta, gamma, R0
    @returns: None
    """
    #Plot True values of dS, and -beta*S*I
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='true dS/dt')
    #plot -beta*S*I shifted by 1 day
    S_pred = np.zeros(len(S))
    S_pred[1:] = -beta[:-1]*S_cum[:-1]*I_cum[:-1]
    plt.plot(S_pred, label='dS using piecewise median beta')
    plt.plot(pd.Series(S).rolling(window=7).mean(), label='Weekly rolling mean of dS/dt')
    plt.axvline(x=int(train*len(S)), color='r', linestyle='--', label='Train-Test Split')
    plt.xlabel('Days')
    plt.ylabel('dS/dt')
    plt.title('True dS/dt vs -beta*S*I')
    plt.legend()
    plt.show()

    #Plot True values of dI, and beta*S*I - gamma*I
    plt.figure(figsize=(10, 6))
    plt.plot(I, label='true dI/dt')
    #plot beta*S*I - gamma*I shifted by 1 day
    I_pred = np.zeros(len(I))
    I_pred[1:] = beta[:-1]*S_cum[:-1]*I_cum[:-1] - gamma[:-1]*I_cum[:-1]
    plt.plot(I_pred, label='dI using piecewise median beta and gamma')
    plt.plot(pd.Series(I).rolling(window=7).mean(), label='Weekly rolling mean of dI/dt')
    plt.axvline(x=int(train*len(S)), color='r', linestyle='--', label='Train-Test Split')
    plt.xlabel('Days')
    plt.ylabel('dI/dt')
    plt.legend()
    plt.title('True dI/dt vs beta*S*I - gamma*I')
    plt.show()

    #Plot True values of dR, and gamma*I
    plt.figure(figsize=(10, 6))
    plt.plot(R, label='true dR/dt')
    #plot gamma*I shifted by 1 day
    R_pred = np.zeros(len(R))
    R_pred[1:] = gamma[:-1]*I_cum[:-1]
    plt.plot(R_pred, label='dR using piecewise median gamma')
    plt.plot(pd.Series(R).rolling(window=7).mean(), label='Weekly rolling mean of dR/dt')
    plt.axvline(x=int(train*len(S)), color='r', linestyle='--', label='Train-Test Split')
    plt.xlabel('Days')
    plt.ylabel('dR/dt')
    plt.legend()
    plt.title('True dR/dt vs gamma*I')
    plt.show()

def plot_beta_gamma(beta, gamma, R0):
    """
    @params: beta, gamma, R0
    @returns: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(beta, label='Beta')
    plt.plot(gamma, label='Gamma')
    # plt.plot(R0, label='R0')
    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.title('SIR Parameters: Beta, Gamma, R0')
    plt.legend()
    plt.show()

def read_data(file):
    """
    @params: None
    @returns: S, I, R, S_cum, I_cum, R_cum
    """
    # Reading data
    df = pd.read_csv(file)
    # df = df[100:200]
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

    extra_elements = len(S) % window
    if extra_elements > 0:
        S = S[:-extra_elements]
        I = I[:-extra_elements]
        R = R[:-extra_elements]
        S_cum = S_cum[:-extra_elements]
        I_cum = I_cum[:-extra_elements]
        R_cum = R_cum[:-extra_elements]
        print(1)

    # Calculating beta and gamma
    beta = calc_beta(S, S_cum, I_cum)
    gamma = calc_gamma(R, I_cum)

    # Printing results
    # print(f'Beta: {beta}')
    # print(f'Gamma: {gamma}')
    # print(f'R0: {R0}')

    # Plotting
    # plot(S, I, R, S_cum, I_cum, R_cum, dS, dI, dR)
    plot_true_beta_gamma(beta, gamma, S, I, R, S_cum, I_cum, R_cum, 1)
    plot_beta_gamma(beta, gamma, beta/gamma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../Data/delhi_2020.csv')
    parser.add_argument('--pop', type=int, default=1e7)
    args = parser.parse_args()
    file = args.file
    N = args.pop
    main(file)