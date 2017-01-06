import datetime as dt
date_format = "%Y-%m-%d"


def tsp_to_datetime(timestamp):
    date = dt.datetime.fromtimestamp(timestamp)
    return date


def tsp_to_date_str(timestamp):
    date = dt.datetime.fromtimestamp(timestamp).strftime(date_format)
    return date


def date_to_tsp(date):
    print("still need to be done ;)")
