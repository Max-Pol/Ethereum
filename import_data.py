import requests
import json
import date_app.dateFormat as df
# import os


def get_data(fsym, tsym, time_range="hours"):
    url = 'https://min-api.cryptocompare.com/data/histo' + time_range[:-1]
    params = {"e": "CCCAGG",
              "useBTC": "true",
              "aggregate": "1",
              "fsym": fsym,
              "tsym": tsym,
              "limit": str(df.time_from_creation(fsym, time_range))}
    response = requests.get(url=url, params=params)  # limit params is 2000 max
    print("Imported from {}".format(url))
    return response.json()


# request data
data = get_data("ETH", "USD", "days")
if data["Response"] != "Success":
    raise ValueError('HTTP Response Error')

# delete zeros at the beginning & first 10 samples
while data['Data'][0]['volumeto'] == 0:
    del data['Data'][0]
for i in range(10):
    del data['Data'][0]

# save data
with open('local_data/data.json', 'w') as f:
    json.dump(data["Data"], f)


# Print data sample
# for daily_data in data["Data"]:
#     date = df.tsp_to_date_str(daily_data['time'])
#     print ('{}: {}'.format(date, daily_data['close']))
