import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the data into a pandas dataframe
data_file = "data.csv"
df = pd.read_csv(data_file)

# Clean the data by removing any missing or null values
df = df.dropna()
df = df.drop(df[df.isin([np.nan, np.inf, -np.inf]).any(1)].index)

# Define the target and feature variables
target = "target"
features = df.drop(target, axis=1)
X = features
y = df[target]

# Standardize the feature data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the hyperparameters for KNeighborsClassifier
min_k = 1
max_k = 31
k_range = list(range(min_k, max_k+1))
weights = ['uniform', 'distance']
metric = ['minkowski', 'euclidean', 'manhattan']

# Define the parameter grid
param_grid = dict(n_neighbors=k_range, weights=weights, metric=metric)

# Initialize the KNeighborsClassifier model
knn = KNeighborsClassifier()

# Use GridSearchCV to find the best hyperparameters
num_folds = 10
grid = GridSearchCV(knn, param_grid, cv=num_folds, scoring='accuracy')
grid.fit(X_scaled, y)

# Print the best parameters and accuracy score from GridSearchCV
print("Best parameters: ", grid.best_params_)
print("Best accuracy score: ", grid.best_score_)

# Use train_test_split to split the data into training and testing sets
test_size = 0.2
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

# Train the KNeighborsClassifier model using the best hyperparameters from GridSearchCV
knn = KNeighborsClassifier(n_neighbors=grid.best_params_["n_neighbors"],
                           weights=grid.best_params_["weights"],
                           metric=grid.best_params_["metric"])
knn.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = knn.predict(X_test)
print("Accuracy score on testing data: ", accuracy_score(y_test, y_pred))
print("Classification report: \n", classification_report(y_test, y_pred))

# Save the model using pickle
model_file = "knn_model.pkl"
pickle.dump(knn, open(model_file, "wb"))
