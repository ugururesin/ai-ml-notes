import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# Define the hyperparameters for SVC
C = [0.1, 1, 10, 100, 1000]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
gamma = [0.1, 1, 10, 100, 1000]

# Define the parameter grid
param_grid = dict(C=C, kernel=kernel, gamma=gamma)

# Initialize the SVC model
svc = SVC()

# Use GridSearchCV to find the best hyperparameters
num_folds = 10
grid = GridSearchCV(svc, param_grid, cv=num_folds, scoring='accuracy')
grid.fit(X_scaled, y)

# Print the best parameters and accuracy score from GridSearchCV
print("Best parameters: ", grid.best_params_)
print("Best accuracy score: ", grid.best_score_)

# Use train_test_split to split the data into training and testing sets
test_size = 0.2
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

# Train the SVC model using the best hyperparameters from GridSearchCV
svc = SVC(C=grid.best_params_["C"],
          kernel=grid.best_params_["kernel"],
          gamma=grid.best_params_["gamma"])
svc.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = svc.predict(X_test)
print("Accuracy score on testing data: ", accuracy_score(y_test, y_pred))
print("Classification report: \n", classification_report(y_test, y_pred))

# Save the model using pickle
model_file = "svc_model.pkl"
pickle.dump(svc, open(model_file, "wb"))
