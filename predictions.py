import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import json
import os


with open('data/data.json', 'r') as f:
    data = json.load(f)


# load the dataset
dataset = []
for daily_data in data:
    dataset.append([daily_data['close']])

# clean zeros at the beginning
i = 0
while dataset[i] == 0:
    i += 1
del dataset[0:i]


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("len(train), len(test) = ({}, {})".format(len(train), len(test)))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)  # dataX (n,1) ; dataY (n,)


# reshape into X=t and Y=t+1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# load or create model
path = 'data/lstm_look_back' + str(look_back) + '.h5'
if os.path.exists(path):
    model = load_model(path)
else:
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_dim=look_back))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=1)
    model.save(path)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# make predictions to several days (sequential predictions)
seqTrain = testX[0].reshape(1, 1, look_back)
seqPredict = model.predict(seqTrain)

for i in range(len(testY) - 1):
    a = np.hstack((seqTrain[-1, 0, 1:look_back],
                   seqPredict[-1, 0])).reshape(1, 1, look_back)
    b = model.predict(a)  # (1,1)
    seqTrain = np.vstack((seqTrain, a))
    seqPredict = np.vstack((seqPredict, b))  # n+1,1

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])  # don't forget []
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])  # don't forget []
seqPredict = scaler.inverse_transform(seqPredict)
# print (seqPredict)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
seqScore = math.sqrt(mean_squared_error(testY[0], seqPredict[:, 0]))
print('Seq Score: %.2f RMSE' % (seqScore))


# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back - 1:len(trainPredict) +
                 look_back - 1, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[train_size + look_back - 1:len(testPredict) +
                train_size + look_back - 1, :] = testPredict

# shift seq predictions for plotting
seqPredictPlot = np.empty_like(dataset)
seqPredictPlot[:, :] = np.nan
start = train_size + look_back - 1
seqPredictPlot[start:start + len(seqPredict), :] = seqPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(seqPredictPlot)
plt.show()
