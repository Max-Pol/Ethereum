import datetime as dt
date_format = "%Y-%m-%d"


def tsp_to_date(timestamp):
    date = dt.datetime.fromtimestamp(timestamp).strftime(date_format)
    return date


def date_to_tsp(date):
    print("still need to be done ;)")
