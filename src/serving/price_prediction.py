"""According to the task in section 'Tankstellen und Vorhersagezeitpunkte' this file implements gas price prediction
for a certain point in time. Only data up to a specified point in time is used for training."""
import argparse
import os
import sys
from decimal import Decimal
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.join(dir_path, '..', '..')
sys.path.insert(0, root)

from src.features.preprocessing import get_datetime_from_string
from src.models.prediction import train_and_predict

OUTPUT_FILE = "price_prediction.csv"


def predict_price(gas_station_id, time, up_to_timestamp):
    model, _, df_forecast = train_and_predict(gas_station_id=gas_station_id, start_time=time, end_time=time,
                                              up_to_timestamp=up_to_timestamp, predict_days=1, use_cached=False,
                                              cache=False)
    deci_cent = round(Decimal(df_forecast.loc[0, 'yhat']), 0)
    print("Predicted for id {} at time {} price {}".format(gas_station_id, time, deci_cent))
    return deci_cent


def predict(data_row):
    index, up_to_timestamp, prediction_timestamp, gas_station_id = data_row
    price = predict_price(gas_station_id, prediction_timestamp, up_to_timestamp)
    return index, price


def predict_prices_in_future(data_rows):
    predictions = [-1] * len(data_rows)
    job_args = []

    for index, data_row in enumerate(data_rows):
        up_to_timestamp, prediction_timestamp, gas_station_id = data_row.split(';')
        up_to_timestamp, prediction_timestamp = get_datetime_from_string(up_to_timestamp), get_datetime_from_string(
            prediction_timestamp)
        gas_station_id = int(gas_station_id)
        job_args.append([index, up_to_timestamp, prediction_timestamp, gas_station_id])

    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(predictions)) as progress_bar:
            for _, result in tqdm(enumerate(p.imap_unordered(predict, job_args))):
                progress_bar.update()
                res_index, res_price = result
                predictions[res_index] = res_price

    with open(OUTPUT_FILE, 'w') as output_file:
        for price, data_row in zip(predictions, data_rows):
            output_file.write("{};{}\n".format(data_row, price))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This script will predict the gas price at given points in time using training data up to an other "
                    "specified point in time.")
    parser.add_argument("--input",
                        type=str, required=True,
                        help="Path to the input data e.g. data/raw/prediction_points.csv")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("The specified file does not exist: {}".format(args.input))
        sys.exit(-1)

    content = open(args.input, 'r').read().splitlines()
    predict_prices_in_future(content)

    print("Successfully wrote price predictions to {}".format(OUTPUT_FILE))
