import pandas as pd

# code to convert xls to csv

xls = pd.read_excel('../Data/measles.xlsx')
# df = xls.parse('Sheet1', index_col=None, na_values=['NA'])
xls.to_csv('../Data/measles.csv', index=False)
