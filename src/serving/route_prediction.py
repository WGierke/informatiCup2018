import os
import sys
from multiprocessing import Pool, cpu_count
from decimal import Decimal
from functools import partial

dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.join(dir_path, '..', '..')
sys.path.insert(0, root)

import datetime as dt
import argparse

import pandas as pd
from src.features.preprocessing import get_datetime_from_string
from shapely.geometry import LineString, MultiPoint
from config import RAW_PATH
from src.fixed_path_gas_station import fixed_path_gas_station as fpgs
from src.models.prediction import train_and_predict
from tqdm import tqdm

### Prepare data, using buffer approach from 3.0-fb-organize_gas_stations.ipynb
GAS_STATIONS_PATH = os.path.join('data', 'raw', 'input_data', 'Eingabedaten', 'Tankstellen.csv')
DEFAULT_ROUTE = "DEFAULT_ROUTE"
OUTPUT_FILE = "route_prediction.csv"

gas_stations = pd.read_csv(GAS_STATIONS_PATH, sep=';',
                           names=['id', 'Name', 'Company', 'Street', 'House_Number', 'Postalcode', 'City', 'Lat',
                                  'Long'],
                           usecols=['id', 'Lat', 'Long', 'Name', 'Street', 'Postalcode', 'City', 'House_Number'],
                           index_col='id')
gas_stations['Adress'] = gas_stations.apply(
    lambda row: '{} {}\n{} {}'.format(row['Street'], row['House_Number'], row['Postalcode'], row['City']), axis=1)
gas_stations.drop(['Street', 'Postalcode', 'City', 'House_Number'], axis=1, inplace=True)


def _hash_point(lat, long):
    # 0.000001 degree can distinguish humans
    # nonetheless, this might cause really weird errors at some point
    return "{0:.7f}".format(round(lat, 7)) + ':' + "{0:.7f}".format(round(long, 7))


gas_station_point_index = gas_stations.copy()
gas_station_point_index['str_pos'] = gas_station_point_index.apply(lambda row: _hash_point(row.Lat, row.Long), axis=1)
gas_station_point_index = gas_station_point_index.reset_index().set_index(['Lat', 'Long'])

gas_station_points = MultiPoint(list(zip(gas_stations['Long'], gas_stations['Lat'])))


def closest_point_on_path(path, point):
    return path.interpolate(path.project(point))


def _get_gas_station_points_near_path(path, radius=0.02):
    return list(path.buffer(radius).intersection(gas_station_points))


def predict_price(gas_station_id, time, in_euros=False):
    """
    Given a gas station id and a timestamp, predict the gas price of the station in the future.
    :param gas_station_id: gas station id
    :param time: timestamp
    :param in_euros: whether to return the price in euros
    :return: price
    """
    model, _, df_forecast = train_and_predict(gas_station_id=gas_station_id, start_time=time, end_time=time, use_cached=True,
                                              cache=True)
    deci_cent = round(Decimal(df_forecast.loc[0, 'yhat']), 0)
    print("Predicted for id {} at time {} price {}".format(gas_station_id, time, deci_cent))
    if in_euros:
        return 0.001 * deci_cent
    return deci_cent


def get_fill_instructions_for_google_path(orig_path, path_length_km, start_time, speed_kmh, capacity_l, start_fuel_l):
    assert start_fuel_l <= capacity_l

    path = LineString([(x, y) for y, x in orig_path])

    # find close gas stations
    positions_df = pd.DataFrame({'orig_position': _get_gas_station_points_near_path(path)})
    positions_df['Lat'] = positions_df['orig_position'].apply(lambda p: p.y)
    positions_df['Long'] = positions_df['orig_position'].apply(lambda p: p.x)
    positions_df = positions_df.set_index(['Lat', 'Long'])
    assert len(positions_df) > 1, 'We want at least one gas station'
    positions_df = positions_df.join(gas_station_point_index)
    positions_df = positions_df.reset_index()
    # approximate them to the path and extimate arrival time
    positions_df['path_position'] = positions_df['orig_position'].apply(lambda p: closest_point_on_path(path, p))
    positions_df['distance_from_start'] = positions_df['orig_position'].apply(
        lambda p: path.project(p, normalized=True) * path_length_km)
    positions_df.sort_values(by='distance_from_start', inplace=True)
    positions_df['stop_time'] = positions_df.distance_from_start.apply(
        lambda dist: start_time + dt.timedelta(hours=dist / speed_kmh))

    # predict the price
    positions_df['price'] = positions_df.apply(lambda row: predict_price(row.id, row.stop_time, in_euros=True),
                                               axis=1)

    # calculate the best filling strategy
    route = pd.DataFrame({
        'cost': positions_df.price,
        'coords': positions_df.path_position.apply(lambda p: fpgs.Coordinate(p.y, p.x))
    })
    result = fpgs.FixedPathGasStation(route, capacity_l, start_fuel_l)
    positions_df['fill_liters'] = result.fill_liters
    positions_df['payment'] = positions_df.price * result.fill_liters
    # positions_df['name'] = positions_df[id]
    stops = positions_df[positions_df.payment != 0]
    return {'start': orig_path[0],
            'end': orig_path[-1],
            'stops': list(stops.orig_position.apply(lambda p: (p.y, p.x)).values),
            'prices': list(stops.price.values),
            'fill_liters': list(stops.fill_liters.values),
            'payment': list(stops.payment.values),
            'address': list(stops.Adress.values),
            'name': list(stops.Name.values),
            'overall_price': result.price}


def predict(index_row, in_euros=False):
    index, row = index_row
    price = predict_price(row['Gas_Station_Id'], get_datetime_from_string(row['Timestamp']), in_euros=in_euros)
    coordinates = fpgs.Coordinate(gas_stations.loc[row['Gas_Station_Id']]['Lat'],
                                  gas_stations.loc[row['Gas_Station_Id']]['Long'])
    return index, price, coordinates


def get_fill_instructions_for_route(f, start_fuel=0, in_euros=False):
    capacity = float(f.readline())
    route = pd.read_csv(f, names=['Timestamp_str', 'Gas_Station_Id'], sep=';')
    route.rename({'Timestamp_str': 'Timestamp'}, axis='columns')
    coordinates = [-1] * len(route)
    cost = [-1] * len(route)

    job_args = [(index, row) for index, row in route.iterrows()]
    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(route)) as pbar:
            for _, result in tqdm(enumerate(p.imap_unordered(partial(predict, in_euros=in_euros), job_args))):
                pbar.update()
                res_index, res_price, res_coordinates = result
                cost[res_index] = res_price
                coordinates[res_index] = res_coordinates

    route['cost'] = cost
    route['coords'] = coordinates
    result = fpgs.FixedPathGasStation(route, capacity, start_fuel)
    # todo join on gas station id for adress
    route['fill_liters'] = result.fill_liters
    route = route.drop(['Timestamp', 'coords'], axis=1)
    route.to_csv(OUTPUT_FILE, sep=';', header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        default=DEFAULT_ROUTE,
                        help="Path to the input data")
    args = parser.parse_args()

    bertha_route = os.path.join(RAW_PATH, "input_data/Eingabedaten/Fahrzeugrouten/Bertha Benz Memorial Route.csv")
    if args.input == DEFAULT_ROUTE:
        print("No route specified. Defaulting to 'Bertha Benz Memorial Route.csv' which is assumed to be in {}".format(
            bertha_route))
        args.input = bertha_route

    if not os.path.isfile(args.input):
        print("The specified file does not exist: {}".format(args.input))
        sys.exit(-1)

    with open(args.input) as f:
        get_fill_instructions_for_route(f, in_euros=False)

    print("Successfully wrote output to {}".format(OUTPUT_FILE))
