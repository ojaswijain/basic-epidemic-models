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
3. Beta/Gamma is constant (Not yet implemented)

Differential Equations:
dS/dt = -beta*S*I
dI/dt = beta*S*I - gamma*I
dR/dt = gamma*I

Available Data:
1. Total Population (N)
2. Newly Infected each day (delta_I)

Parameters to be estimated:
1. Beta
2. Gamma

Major Assumptions:
1. Recovery period is RECOVERY days (Fixed)
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# Defining constants
N = 10000
RECOVERY = 14
epsilon = 1e-7
limit = 0

def gen_R(I):
    """
    @params: I: Total Infected population on each day
    @returns: New Recovered population on each day
    """
    R = np.zeros(len(I))
    R[RECOVERY:] = I[:-RECOVERY]
    return R/N

def fix_I(I, R):
    """
    @params: I: Total Infected population on each day
             R: New Recovered population on each day
    @returns: Net New Infected population on each day, after fixing
    """
    I = I - R*N
    return I/N

def gen_S(I, R):
    """
    @params: I: Infected population on each day
             R: Recovered population on each day
    @returns: Susceptible population on each day
    """
    return  - (I+R)

def cumulative(S, I, R):
    """
    @params: S: Susceptible population on each day
             I: Infected population on each day
             R: Recovered population on each day
    @returns: Cumulative Susceptible, Infected and Recovered population on each day
    """
    R_cum = np.cumsum(R)
    S_cum = 1+np.cumsum(S)
    I_cum = 1-S_cum-R_cum
    return S_cum, I_cum, R_cum

def calc_derivatives(S, I, R):
    """
    @params: S: Cumulative Susceptible population on each day
             I: Cumulative Infected population on each day
             R: Cumulative Recovered population on each day
    @returns: dS/dt, dI/dt, dR/dt for each day, after smoothing
    """

    # Smoothing

    # Calculating derivatives
    dS = np.gradient(S)
    dI = np.gradient(I)
    dR = np.gradient(R)

    return dS, dI, dR

def plot(S, I, R, S_cum, I_cum, R_cum, dS, dI, dR):
    """
    @params: S: Susceptible population on each day
             I: Infected population on each day
             R: Recovered population on each day
             S_cum: Cumulative Susceptible population on each day
             I_cum: Cumulative Infected population on each day
             R_cum: Cumulative Recovered population on each day
    @returns: None
    """
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

    plt.figure(figsize=(10, 6))
    plt.plot(dS, label='Susceptible')
    plt.plot(dI, label='Infected')
    plt.plot(dR, label='Recovered')
    plt.plot(dS+dI+dR, label='Total')
    plt.xlabel('Days')
    plt.ylabel('Derivative of Population')
    plt.title('SIR Model: Derivative of Cases')
    plt.legend()
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
    plt.plot(np.median(R0)*np.ones(len(R0)), label='Median R0')
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
    I = df['Infected'].to_numpy()
    R = gen_R(I)
    I = fix_I(I, R)
    S = gen_S(I, R)
    S_cum, I_cum, R_cum = cumulative(S, I, R)
    return S, I, R, S_cum, I_cum, R_cum

def main():
    """
    @params: None
    @returns: None
    """
    # Reading data
    S, I, R, S_cum, I_cum, R_cum = read_data('../Data/measles.csv')

    # Calculating derivatives
    dS, dI, dR = calc_derivatives(S_cum, I_cum, R_cum)

    # Calculating beta and gamma
    beta = calc_beta(dS, S_cum, I_cum)
    gamma = calc_gamma(dR, I_cum)

    # Calculating R0
    R0 = calc_R0(beta, gamma)

    # Printing results
    # print(f'Beta: {beta}')
    # print(f'Gamma: {gamma}')
    # print(f'R0: {R0}')

    # Plotting
    # plot(S, I, R, S_cum, I_cum, R_cum, dS, dI, dR)
    plot_beta_gamma(beta, gamma, R0)


if __name__ == '__main__':
    main()