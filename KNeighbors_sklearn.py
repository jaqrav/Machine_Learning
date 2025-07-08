import pandas as pd
import numpy as np
import math as m
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def min_max_normalization(x):
    x_min = min(x)
    x_max = max(x)
    return (x - x_min) / (x_max - x_min)

def catch_NaN(v):
    index_of_nan = []
    for i in range(len(v)):
        if m.isnan(v[i]):
            index_of_nan.append(i)
    return np.array(index_of_nan)

# Import data
df = pd.read_csv('https://raw.githubusercontent.com/danielgrijalva/movie-stats/master/movies.csv')

year = df[['year']]
score = df[['score']]
votes = df[['votes']]
budget = df[['budget']]
gross = df[['gross']]
duration = df[['runtime']]

year = min_max_normalization(np.array(year).flatten())
votes = min_max_normalization(np.array(votes).flatten())
budget = min_max_normalization(np.array(budget).flatten())
gross = min_max_normalization(np.array(gross).flatten())
duration = min_max_normalization(np.array(duration).flatten())
score = np.array(score).flatten()

# Handle NaN values
index_of_nan_score = catch_NaN(score)
index_of_nan_votes = catch_NaN(votes)
index_of_nan_budget = catch_NaN(budget)
index_of_nan_gross = catch_NaN(gross)
index_of_nan_duration = catch_NaN(duration)


all_indces = np.concatenate((index_of_nan_score, index_of_nan_votes, index_of_nan_budget,
                             index_of_nan_gross, index_of_nan_duration))
all_indces = np.unique(all_indces)

print('Indices of NaN values:', all_indces)
print(len(all_indces), 'NaN values found in the dataset.')

# Remove rows with NaN values
year = np.delete(year, all_indces)
score = np.delete(score, all_indces)
votes = np.delete(votes, all_indces)
budget = np.delete(budget, all_indces)
gross = np.delete(gross, all_indces)
duration = np.delete(duration, all_indces)

df = pd.DataFrame({
    'year': year,
    'score': score,
    'votes': votes,
    'budget': budget,
    'gross': gross,
    'runtime': duration
})

# Split data into training and testing sets
X = df[['year', 'votes', 'budget', 'gross', 'runtime']].values
y = df['score'].apply(lambda x: 1 if x >= 6 else 0).values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size = 0.8)

# Train KNN classifier using sklearn
fitter = KNeighborsClassifier(n_neighbors = 5)
fitter.fit(xtrain, ytrain)
y_predict_sklearn = fitter.predict(xtest)

print('Accuracy:', '{:.2f}'.format(100*accuracy_score(y_predict_sklearn, ytest)), '%')
print('Precision:', '{:.2f}'.format(100*precision_score(y_predict_sklearn, ytest)), '%')
print('Recall:', '{:.2f}'.format(100*recall_score(y_predict_sklearn, ytest)), '%')
print('F1 Score:', '{:.2f}'.format(100*f1_score(y_predict_sklearn, ytest)), '%')
