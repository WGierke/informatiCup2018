import os
import sys
from unittest import TestCase

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

from src.models.prediction import train_and_predict
from config import MODEL_PATH

GAS_STATION_ID = 1920


class TestPrediction(TestCase):
    def test_model_caching(self):
        model_path = MODEL_PATH.format(GAS_STATION_ID)
        if os.path.isfile(model_path):
            os.remove(model_path)
        model_new, df_future_new, df_forecast_new = train_and_predict(gas_station_id=GAS_STATION_ID, use_cached=True)
        model_cached, df_future_cached, df_forecast_cached = train_and_predict(gas_station_id=GAS_STATION_ID,
                                                                               use_cached=True)
        assert sum(df_forecast_new['yhat'] - df_forecast_cached[
            'yhat']) == 0, "Predictions of freshly trained and serialized model are not equal"
