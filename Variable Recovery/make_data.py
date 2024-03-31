
import pandas as pd
import os

# Load the data
data = pd.read_csv("../Data/covid_19_india.csv")

# Extract unique states
states = data['State'].unique()

# Loop through each state
for state in states:
    filename = "../Data/" + state + '.csv'
    state = state.lower()
    state_year_data = data[(data['State'] == state)]
    state_year_data.to_csv(filename, index=False)
    # Extract unique years for the state
    # years = pd.DatetimeIndex(data[data['State'] == state]['Date']).year.unique()
    # # Loop through each year
    # for year in years:
    #     filename = "../Data/" + state + '_' + str(year) + '.csv'
    #     state_year_data = data[(pd.DatetimeIndex(data['Date']).year == year) & (data['State'] == state)]
    #     state_year_data.to_csv(filename, index=False)
