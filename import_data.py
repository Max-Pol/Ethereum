import requests
import json
# import date_app.dateFormat as df

url = 'https://min-api.cryptocompare.com/data/histoday?' \
      'fsym=ETH&tsym=EUR&limit=200&aggregate=1&e=CCCAGG'
response = requests.get(url=url)
data = response.json()


with open('data/data.json', 'w') as f:
    json.dump(data["Data"], f)

# # Print samples
# if data["Response"] == "Success":
#     for daily_data in data["Data"]:
#         date = df.tsp_to_date(daily_data['time'])
#         print ('{}: {}'.format(date, daily_data['close']))
# else:
#     print ("Error when importing data from the url: ", url)
