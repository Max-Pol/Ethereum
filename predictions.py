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


# PARAMETERS
features_used = [  # comment to discard a feature
    "volumeto",
    "high",
    "low",
    # "time",
    "volumefrom",
    "close",
    "open"
]
look_back = 5
nb_epoch = 100
load_mode = 'load_or_train'
features_plotted = features_used


# print parameters
print("\n*** PARAMETERS ***\n"
      "Number of features: {}\n"
      "Look_back = {}\n"
      "Features plotted: {}"
      .format(len(features_used), look_back, features_plotted))


# load the dataset, with the features in <features_used>
dataset = []
with open('local_data/data.json', 'r') as f:
    data = json.load(f)
for daily_data in data:
    features = []
    for key, value in daily_data.items():
        if key in features_used:
            features.append(value)
    dataset.append(features)


# split into train and test sets
dataset = np.array(dataset)
train_size = int(dataset.shape[0] * 0.67)
test_size = dataset.shape[0] - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("(train, test) = ({}, {})\n".format(len(train), len(test)))


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
test = scaler_test.fit_transform(test)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)
    # dataX: (samples, ts (l_b), features)
    # dataY: (samples, features)


# reshape input to be [samples, time steps, features]
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# load or create model
model_params = {'input_dim': len(features_used)}
model_path = 'local_data/LSTM' + \
             '_inputdim' + str(len(features_used)) + \
             '_l' + str(look_back) + \
             '_epoch' + str(nb_epoch) + '.h5'
if os.path.exists(model_path) and load_mode == 'load_or_train':
    model = load_model(model_path)
else:
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_dim=model_params['input_dim']))
    model.add(Dense(model_params['input_dim']))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit model
    if nb_epoch != 0:
        model.fit(trainX, trainY,
                  nb_epoch=nb_epoch,
                  batch_size=1,
                  verbose=1)

# save model
# model.save(model_path)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler_test.inverse_transform(testPredict)
testY = scaler_test.inverse_transform(testY)

# calculate root mean squared error
print("\n** SCORES **")
for idx, val in enumerate(features_used):
    trainScore = math.sqrt(mean_squared_error(trainY[:, idx],
                                              trainPredict[:, idx]))
    testScore = math.sqrt(mean_squared_error(testY[:, idx],
                                             testPredict[:, idx]))
    print('{} train Score: {:.2f} RMSE'.format(val.upper(), trainScore))
    print('{} test Score: {:.2f} RMSE\n'.format(val.upper(), testScore))


'''
PLOTTING
'''

for feature_plotted in features_plotted:
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset[:, 0])
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back - 1:len(trainPredict) + look_back - 1] \
        = trainPredict[:, features_used.index(feature_plotted)]
    trainPredictPlot = trainPredictPlot.reshape(len(trainPredictPlot), 1)

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset[:, 0])
    testPredictPlot[:] = np.nan
    test_idx = train_size + look_back - 1
    testPredictPlot[test_idx:len(testPredict) + test_idx]  \
        = testPredict[:, features_used.index(feature_plotted)]
    testPredictPlot = testPredictPlot.reshape(len(testPredictPlot), 1)

    # plot baseline and predictions
    plt.plot(dataset[:, features_used.index(feature_plotted)])
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.title(feature_plotted.upper())
    plt.show()
