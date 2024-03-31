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
Implementing in piecewise form
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd

# Defining constants
N = 1e7
RECOVERY = 14
epsilon = 1e-7
limit = 0
train = 0.8
test = 0.2
window = 7

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
    plt.plot(beta*S_cum*I_cum - gamma*I_cum, label='dI using (weighted) median beta and gamma')
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


def opt_beta(beta, S, I):
    
    #split beta, gamma, S, I into chunks of window days
    beta = np.array_split(beta, len(beta)/window)
    S = np.array_split(S, len(S)/window)
    I = np.array_split(I, len(I)/window)

    opt_beta = np.zeros(len(beta)*window)
    for i in range(len(beta)):
            #calculate beta_opt for each chunk
            b = cp.Variable(1)
            objective = cp.Minimize(cp.sum(cp.multiply(cp.abs(beta[i]-b), cp.multiply(S[i],I[i]))))
            constraints = [b >= 0]
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            opt_beta[i*window:(i+1)*window] = b.value
    return opt_beta

def opt_gamma(gamma, I):
    #split beta, gamma, S, I into chunks of window days
    gamma = np.array_split(gamma, len(gamma)/window)
    I = np.array_split(I, len(I)/window)

    opt_gamma = np.zeros(len(gamma)*window)
    for i in range(len(gamma)):
            #calculate gamma_opt for each chunk
            g = cp.Variable(1)
            constraints = [g >= 0]
            objective = cp.Minimize(cp.sum(cp.multiply(cp.abs(gamma[i]-g), I[i])))
            prob = cp.Problem(objective, constraints)
            result = prob.solve()
            opt_gamma[i*window:(i+1)*window] = g.value
    return opt_gamma

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
    beta_opt = opt_beta(beta, S_cum, I_cum)
    gamma_opt = opt_gamma(gamma, I_cum)

    # Plotting
    # plot_true_beta_gamma(beta_opt, gamma_opt, S, I, R, S_cum, I_cum, R_cum, 1)
    plot_beta_gamma(beta_opt, gamma_opt, calc_R0(beta_opt, gamma_opt))

if __name__ == '__main__':
    main()