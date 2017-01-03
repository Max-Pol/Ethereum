import numpy as np
import matplotlib.pyplot as plt
import json
import date_app.dateFormat as df

dates = []
values = []

with open('data/data.json', 'r') as f:
    data = json.load(f)

for daily_data in data:
    print("{}: {}".format(df.tsp_to_date(daily_data['time']),
          daily_data['close']))
    dates.append(daily_data['time'])
    values.append(daily_data['close'])


x = np.array(dates)
y = np.array(values)
plt.plot(x, y)

plt.show()
