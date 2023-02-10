import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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

# Define the hyperparameters for DecisionTreeClassifier
criterion = ['gini', 'entropy']
splitter = ['best', 'random']
max_depth = [int(x) for x in np.linspace(1, 32, num=32)]
min_samples_split = [2, 5, 10, 20]

# Define the parameter grid
param_grid = dict(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split)

# Initialize the DecisionTreeClassifier model
dtc = DecisionTreeClassifier()

# Use GridSearchCV to find the best hyperparameters
num_folds = 10
grid = GridSearchCV(dtc, param_grid, cv=num_folds, scoring='accuracy')
grid.fit(X_scaled, y)

# Print the best parameters and accuracy score from GridSearchCV
print("Best parameters: ", grid.best_params_)
print("Best accuracy score: ", grid.best_score_)

# Use train_test_split to split the data into training and testing sets
test_size = 0.2
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

# Train the DecisionTreeClassifier model using the best hyperparameters from GridSearchCV
dtc = DecisionTreeClassifier(criterion=grid.best_params_["criterion"],
                             splitter=grid.best_params_["splitter"],
                             max_depth=grid.best_params_["max_depth"],
                             min_samples_split=grid.best_params_["min_samples_split"])
dtc.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = dtc.predict(X_test)
print("Accuracy score on testing data: ", accuracy_score(y_test, y_pred))


# Save the model to disk
model_filename = "dtc_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(dtc, file)

# Load the model from disk and make predictions on new data
loaded_model = pickle.load(open(model_filename, 'rb'))
y_pred = loaded_model.predict(X_test)
print("Accuracy score on testing data using loaded model: ", accuracy_score(y_test, y_pred))
