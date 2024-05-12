import pandas as pd

data = pd.read_csv("crime.csv",encoding='latin-1',na_values=["?"])
missing_ratio = data.isnull().mean()
threshold = 0.5
cols_to_drop = missing_ratio[missing_ratio > threshold].index
data2 = data.drop(columns=cols_to_drop)


col1 = data2.columns.get_loc('murders')
col2 = data2.columns.get_loc('arsonsPerPop')

data2 = data2.drop(data2.columns[col1:col2+1], axis=1)
data3 = data2.dropna()

data3.to_csv('crime_vio.csv', index=False)
