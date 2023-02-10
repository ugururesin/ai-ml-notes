import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Define the hyperparameters for RandomForestClassifier
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Define the parameter grid
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap)

# Initialize the RandomForestClassifier model
rfc = RandomForestClassifier()

# Use GridSearchCV to find the best hyperparameters
num_folds = 10
grid = GridSearchCV(rfc, param_grid, cv=num_folds, scoring='accuracy')
grid.fit(X_scaled, y)

# Print the best parameters and accuracy score from GridSearchCV
print("Best parameters: ", grid.best_params_)
print("Best accuracy score: ", grid.best_score_)

# Use train_test_split to split the data into training and testing sets
test_size = 0.2
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

# Train the RandomForestClassifier model using the best hyperparameters from GridSearchCV
rfc = RandomForestClassifier(n_estimators=grid.best_params_["n_estimators"],
                             max_depth=grid.best_params_["max_depth"],
                             min_samples_split=grid.best_params_["min_samples_split"],
                             min_samples_leaf=grid.best_params_["min_samples_leaf"],
bootstrap=grid.best_params_["bootstrap"])
rfc.fit(X_train, y_train)

# Evaluate the RandomForestClassifier model using the testing set
y_pred = rfc.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))

# Save the RandomForestClassifier model using pickle
model_file = "RandomForestClassifier_model.pkl"
with open(model_file, 'wb') as file:
pickle.dump(rfc, file)
