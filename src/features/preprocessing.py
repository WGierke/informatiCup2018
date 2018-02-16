import pytz
from datetime import datetime


def get_datetime_from_string(s):
    return datetime.strptime(s + "00", '%Y-%m-%d %H:%M:%S%z').astimezone(pytz.utc).replace(tzinfo=None)
