import urllib.request
import datetime
import time

for i in range(26, 30):
    for j in [1, 6, 12, 18]:
        date = datetime.datetime(2016, 12, i, j)
        date_timestamp = time.mktime(date.timetuple())
        print ("2015-12-{0} {1}h: {2}".format(i, j, date_timestamp))
        url = 'https://min-api.cryptocompare.com/data/pricehistorical' \
              '?fsym=ETH&tsyms=EUR&ts=' + str(date_timestamp)
        response = urllib.request.urlopen(url)
        print(response.read())
    print ('\n')

# datetime.fromtimestamp(1346236702)
