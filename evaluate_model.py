import joblib
df = pd.read_csv('preprocessed_data.csv', delimiter=';')


gb = joblib.load('model.joblib')
# Evaluate the model on the testing set
score = gb.score(X, y)
print(f'R-squared score on testing set: {score:.3f}')