from datetime import datetime


def get_datetime_from_string(s, keep_utc=False):
    if keep_utc:
        return datetime.strptime(s + "00", '%Y-%m-%d %H:%M:%S%z')
    else:
        return datetime.strptime(s[:s.find('+')], '%Y-%m-%d %H:%M:%S')
