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
    beta = -dS/(I*S)
    #if S or I = 0 then beta = 0
    beta = np.nan_to_num(beta, copy=True, posinf=limit, neginf=-limit)
    # # Replace with piecewise median, piece size = window while keeping array length same
    # num_pieces = len(beta) // window
    # beta_pieces = np.array_split(beta[:num_pieces * window], num_pieces)
    # # Calculate the median for each piece and create a new array repeating medians
    # beta = np.concatenate([np.full(window, np.median(piece)) for piece in beta_pieces])
    return beta

def calc_gamma(dR, I):
    """
    @params: dR, I
    @returns: gamma based on best fit
    """
    gamma = dR/I
    gamma = np.nan_to_num(gamma, copy=True, posinf=limit, neginf=-limit)
    # #Replace with piecewise median, piece size = window while keeping array length same
    # num_pieces = len(gamma) // window
    # gamma_pieces = np.array_split(gamma[:num_pieces * window], num_pieces)
    # # Calculate the median for each piece and create a new array repeating medians
    # gamma = np.concatenate([np.full(window, np.median(piece)) for piece in gamma_pieces])
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
    plt.plot(-beta*S_cum*I_cum, label='dS using piecewise median beta')
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
    plt.plot(beta*S_cum*I_cum - gamma*I_cum, label='dI using piecewise median beta and gamma')
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
    plt.plot(gamma*I_cum, label='dR using piecewise median gamma')
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
    # plt.plot(beta, label='Beta')
    # plt.plot(gamma, label='Gamma')
    plt.plot(R0, label='R0')
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

    #compress the betas and gammas into one value per week
    beta_compressed = beta[::window]
    gamma_compressed = gamma[::window]

    """
    Fit a linear regression model to beta 
    Use the last 5 values of beta to predict the next value
    Train on first 30 values of beta 
    Test on next 8 values of beta 
    """

    #vectors of length 5 for train and test

    train = 0.8
    test = 0.2

    train_x = np.zeros((int(train*len(beta)-10), 10))
    train_y = np.zeros(int(train*len(beta)-10))
    test_x = np.zeros((int(test*len(beta)-10), 10))
    test_y = np.zeros(int(test*len(beta)-10))

    for i in range(0,int(train*len(beta)-10)-10):
        train_x[i] = beta[i:i+10] 
        train_y[i] = beta[i+10]
    
    for i in range(0, int(test*len(beta)-10)-10):
        test_x[i] = beta[i+int(train*len(beta)-10):i+int(train*len(beta)-10)+10]
        test_y[i] = beta[i+int(train*len(beta)-10)+10]

    #train the model

    from sklearn.linear_model import LinearRegression
    
    beta_model = LinearRegression()
    beta_model.fit(train_x, train_y)
    beta_train = beta_model.predict(train_x)
    
    #extrapolate betas
    #use previous predictions to predict next value

    beta_pred = np.zeros(int(test*len(beta)-10))
    beta_pred[0:10] = beta_model.predict(test_x[0:10])
    for i in range(10, int(test*len(beta)-10)):
        beta_pred[i] = beta_model.predict(beta_pred[i-10:i].reshape(1, -1))

    pred = np.concatenate((beta_train, beta_pred))
    true = np.concatenate((train_y, test_y))

    #plot the extrapolated betas
    plt.figure(figsize=(10, 6))
    plt.plot(pred, label='Predicted')
    plt.plot(true, label='True')
    plt.legend()
    plt.show()

    #print the model parameters

    print(beta_model.coef_)
    print(beta_model.intercept_)
    print(beta_model.score(train_x, train_y))
    print(beta_model.score(test_x, test_y))
    
    # plot_true_beta_gamma(beta, gamma, S, I, R, S_cum, I_cum, R_cum, 1)
    # plot_beta_gamma(beta, gamma, beta/gamma)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='../Data/delhi_2020.csv')
    parser.add_argument('--pop', type=int, default=1e7)
    args = parser.parse_args()
    file = args.file
    N = args.pop
    main(file)