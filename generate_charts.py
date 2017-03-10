import numpy as np
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


def plot_features():
    dataset = load_dataset(features_plotted)

    for feature in features_plotted:
        plt.plot(dataset[:, features_plotted.index(feature)])
        plt.title(feature.upper())
        plt.show()


def btc_number(n):
    '''
    Return the total number of bitcoins in existence after the n-th block
    is mined.
    Charateristics:
        - Genesis block: 50 BTC
        - One block is created approximately every 10 minutes, every
          210 000 blocks, the reward is divided by 2.
    '''
    q = np.floor(n / 210000)
    r = n % 210000
    total = 210000 * 50 * (2 - 1 / 2. ** (q - 1)) + r * 50 / 2. ** q
    return total


x = np.linspace(0, 3000000, num=3000)
plt.plot(x, btc_number(x))
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
