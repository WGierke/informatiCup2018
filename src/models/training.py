import datetime
import os
import sys
import pickle

import pandas as pd
from fbprophet import Prophet

# Add parent directory and root directory to PATH
dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.join(dir_path, '..', '..')
sys.path.insert(0, root)
sys.path.append('..')
from config import RAW_PATH, PROCESSED_PATH, MODEL_PATH, GAS_PRICE_PATH, GAS_STATIONS_PATH
from src.features.preprocessing import get_datetime_from_string

DEFAULT_GAS_STATION_ID = 1920
DEFAULT_UP_TO_DAYS = 31


def validate_state(state):
    valid_states = ['BB', 'BE', 'BW', 'BY', 'HB', 'HE', 'HH', 'MV', 'NI', 'NW', 'RP', 'SH', 'SL', 'SN', 'ST', 'TH',
                    'unknown']
    if state not in valid_states:
        raise ValueError("Expected one of {} but got {} instead.".format(valid_states, state))


def get_holidays_df_from_state(state):
    """
    Return the holidays of a given state
    :param state: Abbreviation of the state
    :return: DataFrame containing the holidays starting in 2014
    """
    validate_state(state)
    return pd.read_csv(os.path.join(PROCESSED_PATH, "holidays_{}.csv".format(state)))


def get_vacations_df_from_state(state):
    """
    Return the vacations of a given state
    :param state: Abbreviation of the state
    :return: DataFrame containing the vacations starting in 2014
    """
    validate_state(state)
    return pd.read_csv(os.path.join(PROCESSED_PATH, "vacations_{}.csv".format(state)))


def train(gas_station_id=DEFAULT_GAS_STATION_ID, up_to_days=DEFAULT_UP_TO_DAYS, cache=True):
    """
    Train Prophet on the prices of the given gas station up to a specified amount of days
    :param gas_station_id: Internal identifier of the gas station
    :param up_to_days: Last days that should be excluded from training
    :param cache: Whether to persist the model
    :return: fitted model, DataFrame the model was not fitted to according to up_to_days
    """
    gas_station_path = os.path.join(GAS_PRICE_PATH, "{}.csv".format(gas_station_id))
    # If we're on the CI server, overwrite the path to the specific gas station with a fixed to save bandwidth
    if os.environ.get('CI', False):
        gas_station_path = os.path.join(RAW_PATH, "1920.csv")
    gas_stations_df = pd.read_csv(GAS_STATIONS_PATH, sep=',')
    gas_station_state = gas_stations_df[gas_stations_df["id"] == gas_station_id]["State"].iloc[0]

    df_gas_station = pd.read_csv(gas_station_path, names=['Timestamp', 'Price'], sep=';')
    df_holidays = get_holidays_df_from_state(gas_station_state)
    df_vacations = get_vacations_df_from_state(gas_station_state)

    holidays_df = pd.concat((df_holidays, df_vacations))
    m = Prophet(holidays=holidays_df)
    df_fb = df_gas_station.copy()
    df_fb['y'] = df_fb['Price']
    df_fb['ds'] = df_fb['Timestamp'].apply(lambda x: get_datetime_from_string(str(x), keep_utc=False))
    df_fb.drop(['Timestamp', 'Price'], inplace=True, axis=1)
    if up_to_days > 0:
        start_future = df_fb.iloc[-1, :]['ds'] - datetime.timedelta(days=up_to_days)
        df_past = df_fb[df_fb['ds'] < start_future]
        df_future = df_fb[df_fb['ds'] >= start_future]
    else:
        df_past = df_fb
        df_future = pd.DataFrame(columns=['y'])
    m.fit(df_past)
    if cache:
        pickle.dump(m, open(MODEL_PATH.format(gas_station_id), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return m, df_future
