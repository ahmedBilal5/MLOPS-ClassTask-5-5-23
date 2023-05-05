import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
# Load the Wine Quality dataset

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, delimiter=';')

# Preprocess the data
imp = SimpleImputer(strategy='mean')
df = imp.fit_transform(df)
scaler = StandardScaler()
df = scaler.fit_transform(df)
df.to_csv('preprocessed_data.csv')