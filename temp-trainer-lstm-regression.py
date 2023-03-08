# Importing Libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense

# load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv', usecols=[1])

# assess the data
print(data.head())

# process the data
# split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# normalize the data
mean = train.mean(axis=0)
std = train.std(axis=0)
train = (train - mean) / std
test = (test - mean) / std

# create the dataset for training
def create_dataset(dataset, lookback):
    dataX, dataY = [], []
    for i in range(len(dataset) - lookback - 1):
        a = dataset[i:(i + lookback), 0]
        dataX.append(a)
        dataY.append(dataset[i + lookback, 0])
    return np.array(dataX), np.array(dataY)

lookback = 5
trainX, trainY = create_dataset(train.values, lookback)
testX, testY = create_dataset(test.values, lookback)

# reshape the data for LSTM
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# define the LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(lookback, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# train the model
history = model.fit(trainX, trainY, epochs=100, batch_size=16, validation_data=(testX, testY))

# evaluate the model
scores = model.evaluate(testX, testY)
print('Test loss:', scores)

# make predictions
predictions = model.predict(testX)

# visualize the results
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
