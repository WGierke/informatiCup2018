from datetime import datetime


def get_datetime_from_string(s, keep_utc=True):
    if keep_utc:
        return datetime.strptime(s + "00", '%Y-%m-%d %H:%M:%S%z')
    else:
        return datetime.strptime(s[:-3], '%Y-%m-%d %H:%M:%S')
