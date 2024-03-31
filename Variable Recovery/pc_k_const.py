"""
Code: BTech Project Thesis
Author: Ojaswi Jain
Guide: Prof. Ajit Rajwade
Date: 20th Aug 2023
"""

"""
Convex optimized solution for robust fitting
Let the constant supposed beta be called b0
We need to find b0 such that:
    |dI/dt - b0*S*I| is minimized
which is equivalent to:
    |b_true*S*I - b0*S*I| is minimized
Hence, we need to find b0 such that:
    Summation(|b(t)*S(t)*I(t) - b0*S(t)*I(t)|) is minimized
Which is the same as:
    minSum(|b(t)-b0|*S(t)*I(t))
This is a convex optimization problem

MODIFICATION TO KEEP BETA/GAMMA CONSTANT:
Convert the above problem to:
    minSum(|b(t)-b0|*S(t)*I(t) + |g(t)-k*b0|*I(t))
where k is a constant

Implementation of the above in piecewise form
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from scipy.optimize import minimize

# Defining constants
N = 1e7
RECOVERY = 14
epsilon = 1e-7
limit = 0
train = 0.8
test = 0.2
window = 7
weeks = 42
miss = False
frac_miss = 0.1

def calc_deltas(S, I, R):
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
    plt.plot(-beta*S_cum*I_cum, label='dS using median beta')
    plt.plot(pd.Series(S).rolling(window=7).mean(), label='Weekly rolling mean of dS/dt')
    # plt.axvline(x=int(train*len(S)), color='r', linestyle='--', label='Train-Test Split')
    if miss == True:
        vals = -beta[idx]*S_cum[idx]*I_cum[idx]
        plt.scatter(idx, vals, color='r', label='Missing Values')
    plt.xlabel('Days')
    plt.ylabel('dS/dt')
    plt.title('True dS/dt vs -beta*S*I')
    plt.legend()
    plt.show()

    #Plot True values of dI, and beta*S*I - gamma*I
    plt.figure(figsize=(10, 6))
    plt.plot(I, label='true dI/dt')
    plt.plot(beta*S_cum*I_cum - gamma*I_cum, label='dI using piecewise median beta and gamma, \n piecewise ratio constant')
    plt.plot(pd.Series(I).rolling(window=7).mean(), label='Weekly rolling mean of dI/dt')
    # plt.axvline(x=int(train*len(S)), color='r', linestyle='--', label='Train-Test Split')
    if miss == True:
        vals = beta[idx]*S_cum[idx]*I_cum[idx] - gamma[idx]*I_cum[idx]
        plt.scatter(idx, vals, color='r', label='Missing Values')
    plt.xlabel('Days')
    plt.ylabel('dI/dt')
    plt.legend()
    plt.title('True dI/dt vs beta*S*I - gamma*I')
    plt.show()

    #Plot True values of dR, and gamma*I
    plt.figure(figsize=(10, 6))
    plt.plot(R, label='true dR/dt')
    plt.plot(gamma*I_cum, label='dR using median gamma')
    plt.plot(pd.Series(R).rolling(window=7).mean(), label='Weekly rolling mean of dR/dt')
    # plt.axvline(x=int(train*len(S)), color='r', linestyle='--', label='Train-Test Split')
    if miss == True:
        vals = gamma[idx]*I_cum[idx]
        plt.scatter(idx, vals, color='r', label='Missing Values')
    plt.xlabel('Days')
    plt.ylabel('dR/dt')
    plt.legend()
    plt.title('True dR/dt vs gamma*I')
    plt.show()

def calc_beta(dS, S, I):
    """
    @params: dS/dt, S, I
    @returns: beta based on best fit
    """
    beta = -dS/(I*S)
    #if S or I = 0 then beta = 0
    beta = np.nan_to_num(beta, copy=True, posinf=limit, neginf=-limit)
    return beta

def calc_gamma(dR, I):
    """
    @params: dR, I
    @returns: gamma based on best fit
    """
    gamma = dR/I
    gamma = np.nan_to_num(gamma, copy=True, posinf=limit, neginf=-limit)
    return gamma

def objective_function(b, beta, gamma, S, I, k):
    loss = np.sum(np.abs(beta - b) * S * I + np.abs(gamma - k * b) * I)
    return loss

def opt_beta_k(beta, gamma, S, I):

    k_opt = 0
    loss = np.inf

    best_beta = np.zeros(len(beta))
    best_gamma = np.zeros(len(gamma))

    idx = None
    if miss == True:
        #Make some data as missing
        idx = np.random.choice(len(beta), int(frac_miss*len(beta)), replace=False)
        I[idx] = 0

    #split beta, gamma, S, I into chunks of window days
    beta = np.array_split(beta, len(beta)/window)
    gamma = np.array_split(gamma, len(gamma)/window)
    S = np.array_split(S, len(S)/window)
    I = np.array_split(I, len(I)/window)

    for k in np.arange(0.4, 1, 0.01):

        opt_beta = np.zeros(len(beta)*window)
        opt_gamma = np.zeros(len(gamma)*window)
        k_loss = 0
        for i in range(len(beta)):
            #calculate beta_opt for each chunk
            result = minimize(objective_function, np.median(beta[i]), args=(beta[i], gamma[i], S[i], I[i], k), method='Nelder-Mead')
            opt_beta[i*window:(i+1)*window] = result.x[0]
            opt_gamma[i*window:(i+1)*window] = k*result.x[0]
            k_loss += result.fun

        if k_loss < loss:
            loss = k_loss
            best_beta = opt_beta
            best_gamma = opt_gamma
            k_opt = k

    return best_beta, best_gamma, k_opt, idx

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
    # plt.plot(np.median(beta)*np.ones(len(beta)), label='Median Beta')
    # plt.plot(np.median(gamma)*np.ones(len(gamma)), label='Median Gamma')
    # plt.plot(np.median(R0)*np.ones(len(R0)), label='Median R0')
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
    days = weeks*window
    df = df[:days]
    R_cum = (df["Cured"].to_numpy()+df["Deaths"].to_numpy())/N
    I_cum = df["Confirmed"].to_numpy()/N - R_cum
    S_cum = 1-I_cum-R_cum
    S, I, R = calc_deltas(S_cum, I_cum, R_cum)
    return S, I, R, S_cum, I_cum, R_cum

def main():
    """
    @params: None
    @returns: None
    """
    # Reading data
    S, I, R, S_cum, I_cum, R_cum = read_data('../Data/delhi_2020.csv')

    # Calculating beta and gamma
    beta = calc_beta(S, S_cum, I_cum)
    gamma = calc_gamma(R, I_cum)

    
    beta_opt, gamma_opt, k_opt, idx = opt_beta_k(beta, gamma, S_cum, I_cum)

    print(k_opt)
    # print(beta_opt)
    # print(beta)
    # print(gamma_opt)
    # print(gamma)
    #masking out the beta values that are 0, and respective gamma values
    # idx = beta!=0
    # beta = beta[idx]
    # gamma = gamma[idx]
    # k_dist = gamma/beta

    # plt.hist(k_dist, bins=20)
    # plt.show()

    # print(beta/gamma)

    # Plotting
    plot_true_beta_gamma(beta_opt, gamma_opt, S, I, R, S_cum, I_cum, R_cum, 1, idx)
    plot_beta_gamma(beta_opt, gamma_opt, calc_R0(beta_opt, gamma_opt))

if __name__ == '__main__':
    main()
