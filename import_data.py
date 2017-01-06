import requests
import json
import date_app.dateFormat as df


def get_data(fsym, tsym, time_range):
    if (time_range == "day"):
        url = 'https://min-api.cryptocompare.com/data/histoday?' \
              '&e=CCCAGG&useBTC=true&aggregate=1' \
              '&fsym=' + fsym +\
              '&tsym=' + tsym +\
              '&limit=' + str(df.time_from_creation(fsym, "days"))
    elif (time_range == "hour"):
        url = 'https://min-api.cryptocompare.com/data/histohour?' \
              '&e=CCCAGG&useBTC=true&aggregate=1' \
              '&fsym=' + fsym +\
              '&tsym=' + tsym +\
              '&limit=' + str(df.time_from_creation(fsym, "hours"))  # 2000 MAX

    response = requests.get(url=url)
    return response.json()


data = get_data("ETH", "USD", "hour")
with open('data/data.json', 'w') as f:
    json.dump(data["Data"], f)

# # Print samples
# if data["Response"] == "Success":
#     for daily_data in data["Data"]:
#         date = df.tsp_to_date(daily_data['time'])
#         print ('{}: {}'.format(date, daily_data['close']))
# else:
#     print ("Error when importing data from the url: ", url)
