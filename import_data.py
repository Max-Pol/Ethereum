import datetime as dt
import requests

date_format = "%Y-%m-%d"

url = 'https://min-api.cryptocompare.com/data/histoday?' \
      'fsym=ETH&tsym=EUR&limit=10&aggregate=1&e=CCCAGG'
response = requests.get(url=url)
data = response.json()

for daily_data in data["Data"]:
    current_date = dt.datetime.fromtimestamp(
        daily_data['time']).strftime(date_format)
    print ('{}: {}'.format(current_date, daily_data['close']))
