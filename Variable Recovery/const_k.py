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

def calc_deltas(S, I, R):
    dS = np.zeros(len(S))
    dI = np.zeros(len(I))
    dR = np.zeros(len(R))
    dS[1:] = S[1:] - S[:-1]
    dI[1:] = I[1:] - I[:-1]
    dR[1:] = R[1:] - R[:-1]
    return dS, dI, dR

def plot_true_beta_gamma(beta, gamma, S, I, R, S_cum, I_cum, R_cum, train):
    """
    @params: beta, gamma, R0
    @returns: None
    """
    #Plot True values of dS, and -beta*S*I
    plt.figure(figsize=(10, 6))
    plt.plot(S, label='true dS/dt')
    plt.plot(-beta*S_cum*I_cum, label='dS using median beta')
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
    plt.plot(beta*S_cum*I_cum - gamma*I_cum, label='dI using median beta and gamma, ratio constant')
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
    plt.plot(gamma*I_cum, label='dR using median gamma')
    plt.plot(pd.Series(R).rolling(window=7).mean(), label='Weekly rolling mean of dR/dt')
    plt.axvline(x=int(train*len(S)), color='r', linestyle='--', label='Train-Test Split')
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
    b_opt = 0
    loss = np.inf
    for k in np.arange(0.4, 0.6, 0.01):

        result = minimize(objective_function, x0=0, args=(beta, gamma, S, I, k), bounds=[(0, None)])
        
        if result.success and result.fun < loss:
            loss = result.fun
            b_opt = result.x[0]
            k_opt = k
    
    return b_opt, k_opt

# def opt_beta_k(beta, gamma, S, I):
#     """
#     @params: gamma, I
#     @returns: gamma based on convex optimization
#     """
#     k_opt = 0
#     loss = np.inf
#     for k in np.arange(0, 5, 0.1):   
#         b = cp.Variable(1)
#         objective = cp.Minimize(cp.sum((cp.abs(beta-b)*S*I) + cp.abs(gamma-k*b)*I))
#         constraints = [b >= 0]
#         prob = cp.Problem(objective, constraints)
#         result = prob.solve()
#         if b.value is not None and prob.value < loss:
#             k_opt = prob.value
#             b_opt = b.value
#     return b_opt, k_opt

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
    # plt.plot(beta, label='Beta')
    # plt.plot(gamma, label='Gamma')
    plt.plot(R0, label='R0')
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
    I_cum = df["Confirmed"].to_numpy()/N
    R_cum = (df["Cured"].to_numpy()+df["Deaths"].to_numpy())/N
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

    
    beta_opt, k_opt = opt_beta_k(beta, gamma, S_cum, I_cum)
    gamma_opt = k_opt*beta_opt

    print(k_opt)
    print(beta_opt)
    print(beta)
    print(gamma_opt)
    print(gamma)
    # print(gamma/beta)

    # Plotting
    plot_true_beta_gamma(beta_opt, gamma_opt, S, I, R, S_cum, I_cum, R_cum, 1)
    plot_beta_gamma(beta, gamma, gamma/beta)

if __name__ == '__main__':
    main()
