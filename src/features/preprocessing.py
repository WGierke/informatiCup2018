import pytz
from datetime import datetime


def get_datetime_from_string(s):
    """
    Given a datetime string that is converted like described in the task, normalize it to UTC and remove the
    offset information.
    """
    return datetime.strptime(s + "00", '%Y-%m-%d %H:%M:%S%z').astimezone(pytz.utc).replace(tzinfo=None)
