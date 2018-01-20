import datetime
import os
import pickle
import pandas as pd

from .training import DEFAULT_UP_TO_DAYS, DEFAULT_GAS_STATION_ID, train
from config import MODEL_PATH


def predict(model, up_to_days=DEFAULT_UP_TO_DAYS, start_time=None, end_time=None, plot=False):
    """
    Predict the next up_to_days days using the given model
    :param model: Trained Prophet model
    :param up_to_days: Amount of days that should be predicted, ignored if start_time and end_time are not None
    :param start_time: Timestamp of the beginning of the forecast
    :param end_time: Timestamp of the end of the forecast
    :param plot: Whether to plot the forecast
    :return: DataFrame containing the predicted prices
    """
    if start_time is None and end_time is None:
        df_future = model.make_future_dataframe(periods=24 * up_to_days, freq='H')
        start_future = df_future.iloc[-1, :]['ds'] - datetime.timedelta(days=up_to_days)
        df_future = df_future[df_future['ds'] >= start_future]
    elif start_time is not None and end_time is not None:
        indices = pd.date_range(start_time, end_time, freq='H')
        assert len(indices) > 0, "Indices should not be empty"
        df_future = pd.DataFrame(columns=['ds', 'y'])
        df_future['ds'] = indices
    else:
        raise ValueError("Either up_to_days or start_time and end_time must be set appropriately.")
    df_forecast = model.predict(df_future)
    if plot:
        model.plot(df_forecast)
        model.plot_components(df_forecast)
    return df_forecast


def train_and_predict(gas_station_id=DEFAULT_GAS_STATION_ID, start_time=None, end_time=None,
                      up_to_days=DEFAULT_UP_TO_DAYS, plot=False, use_cached=False):
    """
    Train the model for gas_station_id and return the prediction for the next up_to_days days
    :param gas_station_id: Internal identifier of the gas station
    :param up_to_days: Amount of days that should be predicted, ignored if start_time and end_time are not None
    :param start_time: Timestamp of the beginning of the forecast
    :param end_time: Timestamp of the end of the forecast
    :param plot: Whether to plot the forecast
    :param use_cached: Whether to load the serialized model if it exists
    :return: Fitted Model,
             DataFrame containing the true future prices
             DataFrame containing the predicted prices
    """
    model_loaded = False
    if use_cached:
        model_path = MODEL_PATH.format(gas_station_id)
        try:
            if not os.path.isfile(model_path):
                raise ValueError("No model was found at {}".format(model_path))

            model = pickle.load(open(model_path, "rb"))
            df_future = None
            model_loaded = True
        except Exception as e:
            print(e)

    if not model_loaded:
        model, df_future = train(gas_station_id=gas_station_id, up_to_days=up_to_days)
    df_forecast = predict(model, start_time=start_time, end_time=end_time, up_to_days=up_to_days, plot=plot)
    return model, df_future, df_forecast
