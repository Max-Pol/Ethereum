import numpy as np
import matplotlib.pyplot as plt
import json
import date_app.dateFormat as df
import matplotlib.dates as mdates

dates = []
values = []

months = mdates.MonthLocator()
fig, ax = plt.subplots()

with open('data/data.json', 'r') as f:
    data = json.load(f)

for daily_data in data:
    print("{}: {}".format(df.tsp_to_date_str(daily_data['time']),
          daily_data['close']))
    # dates.append(df.tsp_to_date(daily_data['time']))
    dates.append(df.tsp_to_datetime(daily_data['time']))
    values.append(daily_data['close'])


t = np.array(dates)
y = np.array(values)
plt.plot(t, y)

plt.show()
