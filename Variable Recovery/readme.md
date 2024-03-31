Code: BTech Project Thesis
Author: Ojaswi Jain
Guide: Prof. Ajit Rajwade
Date: 20th Aug 2023

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

