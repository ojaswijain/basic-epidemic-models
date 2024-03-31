import pandas as pd

#code to select only rows with State = "Delhi"

df = pd.read_csv("../Data/covid_19_india.csv")
df = df[df['State'] == 'Delhi']
#split by dates into 2020 and 2021
df_2020 = df[df['Date'].str.contains('2020')]
df_2021 = df[df['Date'].str.contains('2021')]
#save both
df_2020.to_csv("../Data/delhi_2020.csv")
df_2021.to_csv("../Data/delhi_2021.csv")
