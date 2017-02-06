# import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import date_app.dateFormat as df
from utils import load_dataset


# PARAMETERS
features_plotted = [  # comment to discard a feature
    "volumeto",
    "high",
    "low",
    # "time",
    "volumefrom",
    "close",
    "open"
]

dataset = load_dataset(features_plotted)

for feature in features_plotted:
    plt.plot(dataset[:, features_plotted.index(feature)])
    plt.title(feature.upper())
    plt.show()

'''
dates = []
values = []
months = mdates.MonthLocator()
fig, ax = plt.subplots()

for daily_data in data:
    # print("{}: {}".format(df.tsp_to_date_str(daily_data['time']),
    #       daily_data['close']))
    # dates.append(df.tsp_to_date(daily_data['time']))
    dates.append(df.tsp_to_datetime(daily_data['time']))
    values.append(daily_data['close'])
'''
