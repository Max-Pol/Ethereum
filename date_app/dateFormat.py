import datetime as dt
date_format = "%Y-%m-%d"
start_date = {}

today = dt.datetime.now()
start_date["ETH"] = dt.datetime(2015, 7, 30)  # estimate
start_date["BTC"] = dt.datetime(2009, 1, 3)  # estimate


def tsp_to_datetime(timestamp):
    date = dt.datetime.fromtimestamp(timestamp)
    return date


def tsp_to_date_str(timestamp):
    date = dt.datetime.fromtimestamp(timestamp).strftime(date_format)
    return date


def date_to_tsp(date):
    print("still need to be done ;)")


def time_from_creation(symbol, format):
    timedelta = today - start_date[symbol]
    if (format == "days"):
        return (timedelta.days)
    elif (format == "hours"):  # timedelta only has the days and seconds fields
        hours_number = timedelta.days * 24 + timedelta.seconds / 3600
        return (int(hours_number))
    elif (format == "minutes"):
        minutes_number = timedelta.days * 24 * 60 + timedelta.seconds / 60
        return (int(minutes_number))
    else:
        raise ValueError('wrong format in method time_from_creation')
