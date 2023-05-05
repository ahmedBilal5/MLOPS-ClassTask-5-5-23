import joblib
import pandas as pd

df = pd.read_csv('preprocessed_data.csv', delimiter=';')
gb = joblib.load('model.joblib')
X = df.drop(columns=['quality'])
y = df['quality']
# Evaluate the model on the entire set
score = gb.score(X, y)
print(f'R-squared score on testing set: {score:.3f}')