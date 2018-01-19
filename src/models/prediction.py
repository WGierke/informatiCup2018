import datetime

from .training import DEFAULT_UP_TO_DAYS, DEFAULT_GAS_STATION_ID, train


def predict(model, up_to_days=DEFAULT_UP_TO_DAYS, plot=False):
    """
    Predict the next up_to_days days using the given model
    :param model: Trained Prophet model
    :param up_to_days: Amount of days that should be predicted
    :param plot: Whether to plot the forecast
    :return: DataFrame containing the predicted prices
    """
    future = model.make_future_dataframe(periods=24 * up_to_days, freq='H')
    forecast = model.predict(future)
    if plot:
        model.plot(forecast)
        model.plot_components(forecast)
    start_future = forecast.iloc[-1, :]['ds'] - datetime.timedelta(days=up_to_days)
    df_future = forecast[forecast['ds'] >= start_future]
    return df_future


def train_and_predict(gas_station_id=DEFAULT_GAS_STATION_ID, up_to_days=DEFAULT_UP_TO_DAYS, plot=False):
    """
    Train the model for gas_station_id and return the prediction for the next up_to_days days
    :param gas_station_id: Internal identifier of the gas station
    :param up_to_days: Amount of days that should be predicted
    :param plot: Whether to plot the forecast
    :return: Fitted Model,
             DataFrame containing the true future,
             DataFrame containing the predicted prices,
    """
    model, df_future = train(gas_station_id=gas_station_id, up_to_days=up_to_days)
    df_forecast = predict(model, up_to_days=up_to_days, plot=plot)
    return model, df_future, df_forecast
