import pandas as pd
import datetime as dt
import os
from shapely.geometry import LineString, MultiPoint

import sys

sys.path.append('..')
from fixed_path_gas_station import fixed_path_gas_station as fpgs
from flask import Flask
app = Flask(__name__)

### Prepare data, using buffer approach from 3.0-fb-organize_gas_stations.ipynb
GAS_STATIONS_PATH = os.path.join('..', '..', 'data', 'raw', 'input_data', 'Eingabedaten', 'Tankstellen.csv')

gas_stations = pd.read_csv(GAS_STATIONS_PATH, sep=';',
                           names=['id', 'Name', 'Company', 'Street', 'House_Number', 'Postalcode', 'City', 'Lat',
                                  'Long'], usecols=['id', 'Lat', 'Long'], index_col='id')


def _hash_point(lat, long):
    # 0.000001 degree can distinguish humans
    # nonetheless, this might cause really weird errors at some point
    return "{0:.7f}".format(round(lat, 7)) + ':' + "{0:.7f}".format(round(long, 7))


gas_station_point_index = gas_stations.copy()
gas_station_point_index['str_pos'] = gas_station_point_index.apply(lambda row: _hash_point(row.Lat, row.Long), axis=1)
gas_station_point_index = gas_station_point_index.reset_index().set_index('str_pos')

gas_station_points = MultiPoint(list(zip(gas_stations['Long'], gas_stations['Lat'])))


def closest_point_on_path(path, point):
    return path.interpolate(path.project(point))


def _get_gas_station_points_near_path(path, radius=0.02):
    return list(path.buffer(radius).intersection(gas_station_points))


def predict_price(id, time):
    # TODO models will be called here
    return 1.30

@app.route("/fill_prediction/google/")
def get_fill_instructions_for_google_path(orig_path, path_length_km, start_time, speed_kmh, capacity_l, start_fuel_l):
    assert start_fuel_l <= capacity_l

    path = LineString([(x, y) for y, x in orig_path])

    # find close gas stations
    positions_df = pd.DataFrame({'orig_position': _get_gas_station_points_near_path(path)})
    assert len(positions_df) > 1, 'We want at least one gas station'

    positions_df['id'] = \
        gas_station_point_index.loc[positions_df['orig_position'].apply(lambda p: _hash_point(p.y, p.x))]['id'].values

    # approximate them to the path and extimate arrival time
    positions_df['path_position'] = positions_df['orig_position'].apply(lambda p: closest_point_on_path(path, p))
    positions_df['distance_from_start'] = positions_df['orig_position'].apply(
        lambda p: path.project(p, normalized=True) * path_length_km)
    positions_df.sort_values(by='distance_from_start', inplace=True)
    positions_df['stop_time'] = positions_df.distance_from_start.apply(
        lambda dist: start_time + dt.timedelta(hours=dist / speed_kmh))

    # predict the price
    positions_df['price'] = positions_df.apply(lambda row: predict_price(row.id, row.stop_time),
                                               axis=1)

    # calculate the best filling strategy
    route = pd.DataFrame({
        'cost': positions_df.price,
        'coords': positions_df.path_position.apply(lambda p: fpgs.Coordinate(p.y, p.x))
    })
    result = fpgs.FixedPathGasStation(route, capacity_l, start_fuel_l)
    positions_df['fill_liters'] = result.fill_liters
    positions_df['payment'] = positions_df.price * result.fill_liters
    stops = positions_df[positions_df.payment != 0]
    return {'msg': 'Have a good time!',
            'start': orig_path[0],
            'end': orig_path[-1],
            'stops': list(stops.orig_position.apply(lambda p: (p.y, p.x)).values),
            'prices': list(stops.price.values),
            'fill_liters': list(stops.fill_liters.values),
            'payment': list(stops.payment.values),
            'overall_price': result.price}

@app.route('/test/')
def test():
    return {'hi':42}


@app.route("/fill_prediction/berta/")
def get_fill_instructions_for_route(path_to_file, start_fuel=0):
    with open(path_to_file, 'r') as f:
        capacity = float(f.readline())
    route = pd.read_csv(path_to_file, names=['Timestamp_str', 'Gas_Station_Id'], sep=';', skiprows=1)
    route['Timestamp'] = route['Timestamp_str'].apply(lambda x: pd.Timestamp(x))
    cost = []
    coordinates = []
    for index, row in route.iterrows():
        cost.append(predict_price(row['Gas_Station_Id'], row['Timestamp']))
        coordinates.append(fpgs.Coordinate(gas_stations.loc[row['Gas_Station_Id']]['Lat']
                                           , gas_stations.loc[row['Gas_Station_Id']]['Long']))
    route['cost'] = cost
    route['coords'] = coordinates
    result = fpgs.FixedPathGasStation(route, capacity, start_fuel)

    route['fill_liters'] = result.fill_liters
    route['payment'] = route.fill_liters * route.cost
    stops = route[route.fill_liters != 0]

    return {'msg': 'not yet implemented',
            'start': tuple(route.iloc[0]['coords']),
            'end': tuple(route.iloc[-1]['coords']),
            'stops': list(stops.coords.apply(lambda coord: tuple(coord)).values),
            'prices': list(stops.cost.values),
            'fill_liters': list(stops.fill_liters.values),
            'payment': list(stops.payment.values),
            'overall_price': result.price}


if __name__ == '__main__':
    print('Potsdam Route')
    path_potsdam_berlin = [(52.390530000000005, 13.064540000000001), (52.39041, 13.065890000000001),
                           (52.39025, 13.06723), (52.39002000000001, 13.068810000000001),
                           (52.389970000000005, 13.069350000000002), (52.38998, 13.06948),
                           (52.389860000000006, 13.07028), (52.38973000000001, 13.07103), (52.38935000000001, 13.07352),
                           (52.3892, 13.07463), (52.38918, 13.075120000000002), (52.389210000000006, 13.07553),
                           (52.389300000000006, 13.0759), (52.3894, 13.076130000000001), (52.389520000000005, 13.07624),
                           (52.38965, 13.07638), (52.389880000000005, 13.0767),
                           (52.390100000000004, 13.077110000000001), (52.390330000000006, 13.077770000000001),
                           (52.390440000000005, 13.078660000000001), (52.39052, 13.079400000000001),
                           (52.390570000000004, 13.08004), (52.39056000000001, 13.08037), (52.390550000000005, 13.0806),
                           (52.390530000000005, 13.080990000000002), (52.390420000000006, 13.083100000000002),
                           (52.390440000000005, 13.083400000000001), (52.39038000000001, 13.083430000000002),
                           (52.39011000000001, 13.0836), (52.38853, 13.084660000000001), (52.38801, 13.0851),
                           (52.38774, 13.085410000000001), (52.38754, 13.085730000000002),
                           (52.38729000000001, 13.086300000000001), (52.38689, 13.087610000000002),
                           (52.386500000000005, 13.088960000000002), (52.38611, 13.09026),
                           (52.38602, 13.090700000000002), (52.3858, 13.09121),
                           (52.385290000000005, 13.092300000000002), (52.38477, 13.09331),
                           (52.384040000000006, 13.094650000000001), (52.383500000000005, 13.095670000000002),
                           (52.38302, 13.096580000000001), (52.37538000000001, 13.110970000000002),
                           (52.37485, 13.112020000000001), (52.37471000000001, 13.112340000000001),
                           (52.37436, 13.113220000000002), (52.373990000000006, 13.114300000000002),
                           (52.37379000000001, 13.11494), (52.373580000000004, 13.11578), (52.37304, 13.11809),
                           (52.37266, 13.119740000000002), (52.37252, 13.120540000000002),
                           (52.37238000000001, 13.121540000000001), (52.37227000000001, 13.122710000000001),
                           (52.37225, 13.12311), (52.372220000000006, 13.12376),
                           (52.372220000000006, 13.124830000000001), (52.372260000000004, 13.128100000000002),
                           (52.37229000000001, 13.131340000000002), (52.37234, 13.1369), (52.37232, 13.13785),
                           (52.37228, 13.13859), (52.37220000000001, 13.13958), (52.37216, 13.140500000000001),
                           (52.372150000000005, 13.141950000000001), (52.37218000000001, 13.14399),
                           (52.37228, 13.147120000000001), (52.3723, 13.14906), (52.37232, 13.151140000000002),
                           (52.37228, 13.15149), (52.37225, 13.151850000000001), (52.37219, 13.152070000000002),
                           (52.372130000000006, 13.152210000000002), (52.372040000000005, 13.152360000000002),
                           (52.371930000000006, 13.15248), (52.37181, 13.152560000000001),
                           (52.37167, 13.152600000000001), (52.37153000000001, 13.152600000000001),
                           (52.3714, 13.152550000000002), (52.371300000000005, 13.15248), (52.3712, 13.152370000000001),
                           (52.37106000000001, 13.152130000000001), (52.37098, 13.151840000000002),
                           (52.37095000000001, 13.151560000000002), (52.370960000000004, 13.15136),
                           (52.371, 13.151090000000002), (52.37109, 13.150830000000001), (52.3712, 13.15066),
                           (52.37129, 13.15056), (52.371460000000006, 13.15046), (52.37163, 13.150430000000002),
                           (52.37181, 13.150400000000001), (52.37322, 13.150360000000001),
                           (52.373670000000004, 13.150350000000001), (52.37375, 13.15032),
                           (52.37451, 13.150310000000001), (52.375710000000005, 13.15028),
                           (52.37670000000001, 13.150250000000002), (52.376960000000004, 13.150250000000002),
                           (52.37715000000001, 13.150220000000001), (52.37742, 13.150160000000001),
                           (52.377720000000004, 13.15013), (52.378040000000006, 13.150120000000001),
                           (52.37812, 13.15009), (52.37825, 13.15004), (52.378800000000005, 13.15004),
                           (52.379270000000005, 13.15009), (52.37962, 13.150150000000002),
                           (52.380010000000006, 13.150240000000002), (52.380370000000006, 13.150360000000001),
                           (52.380990000000004, 13.150620000000002), (52.38165000000001, 13.15098),
                           (52.383500000000005, 13.152170000000002), (52.38440000000001, 13.15277),
                           (52.3858, 13.153670000000002), (52.387080000000005, 13.1545), (52.38745, 13.154760000000001),
                           (52.38768, 13.15496), (52.38794000000001, 13.155190000000001),
                           (52.388380000000005, 13.155660000000001), (52.38891, 13.156350000000002),
                           (52.38927, 13.156920000000001), (52.38965, 13.15755), (52.38984000000001, 13.15792),
                           (52.39011000000001, 13.158520000000001), (52.390460000000004, 13.15943),
                           (52.39074, 13.160380000000002), (52.392900000000004, 13.169300000000002),
                           (52.39408, 13.1742), (52.39439, 13.175370000000001),
                           (52.394830000000006, 13.176800000000002), (52.395320000000005, 13.17805),
                           (52.39578, 13.179070000000001), (52.39621, 13.17993), (52.39678000000001, 13.18092),
                           (52.39714000000001, 13.18148), (52.3975, 13.181970000000002),
                           (52.398340000000005, 13.183000000000002), (52.39922000000001, 13.184000000000001),
                           (52.399530000000006, 13.18438), (52.40012, 13.18504),
                           (52.400940000000006, 13.185910000000002), (52.40171, 13.186750000000002),
                           (52.402260000000005, 13.187420000000001), (52.403830000000006, 13.18917),
                           (52.407830000000004, 13.193690000000002), (52.40982, 13.19593),
                           (52.410230000000006, 13.19631), (52.41085, 13.19678),
                           (52.411280000000005, 13.197030000000002), (52.41158000000001, 13.197180000000001),
                           (52.41223, 13.197420000000001), (52.412620000000004, 13.197510000000001),
                           (52.413030000000006, 13.19757), (52.413880000000006, 13.19757),
                           (52.41407, 13.197560000000001), (52.41452, 13.197470000000001),
                           (52.41536000000001, 13.19729), (52.41561, 13.197210000000002),
                           (52.416720000000005, 13.19697), (52.417570000000005, 13.196760000000001),
                           (52.41827000000001, 13.196610000000002), (52.42042000000001, 13.196130000000002),
                           (52.4217, 13.195850000000002), (52.422740000000005, 13.19561),
                           (52.423030000000004, 13.195500000000001), (52.42322000000001, 13.195390000000002),
                           (52.423410000000004, 13.195260000000001), (52.42360000000001, 13.195120000000001),
                           (52.42381, 13.194930000000001), (52.42409000000001, 13.194640000000001),
                           (52.42443, 13.194170000000002), (52.424820000000004, 13.1935),
                           (52.425160000000005, 13.19293), (52.42549, 13.192450000000001),
                           (52.425720000000005, 13.192160000000001), (52.42607, 13.191820000000002),
                           (52.426300000000005, 13.191640000000001), (52.42649, 13.19152),
                           (52.42685, 13.191350000000002), (52.427310000000006, 13.191230000000001),
                           (52.427530000000004, 13.191210000000002), (52.427890000000005, 13.191230000000001),
                           (52.42887, 13.191460000000001), (52.43121000000001, 13.19204),
                           (52.43244000000001, 13.192340000000002), (52.43292, 13.19246), (52.433400000000006, 13.1926),
                           (52.43365000000001, 13.19269), (52.43403000000001, 13.192870000000001),
                           (52.434470000000005, 13.193150000000001), (52.43478, 13.19339),
                           (52.43506000000001, 13.193650000000002), (52.435340000000004, 13.19396),
                           (52.43573000000001, 13.194440000000002), (52.43797000000001, 13.197270000000001),
                           (52.438610000000004, 13.198080000000001), (52.44021000000001, 13.2001), (52.44169, 13.20198),
                           (52.44489, 13.206010000000001), (52.446180000000005, 13.207640000000001),
                           (52.45031, 13.212860000000001), (52.47092000000001, 13.238930000000002),
                           (52.472350000000006, 13.240730000000001), (52.47289000000001, 13.24136),
                           (52.474680000000006, 13.243440000000001), (52.47838, 13.247610000000002),
                           (52.48109, 13.250670000000001), (52.48225000000001, 13.25201), (52.482800000000005, 13.2527),
                           (52.48602, 13.25679), (52.48906, 13.260610000000002), (52.491670000000006, 13.26392),
                           (52.49271, 13.26524), (52.49497, 13.268040000000001),
                           (52.495160000000006, 13.268360000000001), (52.495760000000004, 13.26917),
                           (52.496280000000006, 13.26984), (52.497170000000004, 13.27105),
                           (52.497840000000004, 13.27194), (52.49857, 13.272870000000001),
                           (52.49895000000001, 13.273460000000002), (52.49916, 13.273930000000002),
                           (52.49929, 13.27434), (52.499390000000005, 13.274840000000001),
                           (52.499460000000006, 13.275440000000001), (52.49949, 13.275970000000001),
                           (52.49956, 13.277550000000002), (52.49963, 13.27838), (52.49969, 13.278830000000001),
                           (52.499770000000005, 13.27918), (52.499900000000004, 13.279630000000001),
                           (52.500060000000005, 13.28002), (52.500220000000006, 13.280330000000001),
                           (52.50027000000001, 13.28035), (52.500370000000004, 13.28049),
                           (52.50054, 13.280690000000002), (52.5007, 13.28082), (52.50085000000001, 13.280880000000002),
                           (52.501020000000004, 13.2809), (52.50117, 13.280880000000002),
                           (52.50155, 13.280740000000002), (52.50173, 13.280690000000002),
                           (52.501960000000004, 13.28068), (52.502210000000005, 13.280780000000002),
                           (52.502390000000005, 13.28086), (52.503310000000006, 13.28194),
                           (52.50368, 13.282330000000002), (52.503930000000004, 13.282520000000002),
                           (52.50423000000001, 13.28269), (52.504560000000005, 13.28279),
                           (52.50522, 13.282820000000001), (52.50553000000001, 13.28284),
                           (52.50583, 13.282890000000002), (52.50598, 13.282940000000002),
                           (52.506350000000005, 13.283100000000001), (52.506620000000005, 13.28326),
                           (52.508250000000004, 13.284370000000001), (52.509620000000005, 13.28527),
                           (52.51070000000001, 13.28592), (52.511100000000006, 13.286100000000001),
                           (52.511210000000005, 13.286150000000001), (52.51158, 13.286230000000002),
                           (52.511700000000005, 13.286380000000001), (52.511810000000004, 13.286420000000001),
                           (52.51239, 13.28658), (52.512570000000004, 13.28668), (52.512800000000006, 13.28687),
                           (52.5129, 13.286890000000001), (52.51297, 13.286890000000001), (52.51299, 13.28706),
                           (52.51301, 13.28738), (52.51308, 13.28842), (52.51274, 13.288520000000002),
                           (52.51194, 13.288760000000002), (52.511300000000006, 13.288960000000001),
                           (52.510560000000005, 13.289200000000001), (52.510380000000005, 13.289240000000001),
                           (52.51043000000001, 13.289950000000001), (52.510510000000004, 13.291240000000002),
                           (52.51066, 13.293750000000001), (52.51122, 13.30202), (52.51147, 13.30563),
                           (52.51184000000001, 13.31169), (52.512080000000005, 13.315150000000001),
                           (52.51239, 13.320010000000002), (52.51241, 13.320640000000001), (52.51234, 13.32089),
                           (52.512280000000004, 13.320950000000002), (52.51218, 13.321090000000002),
                           (52.51207, 13.32136), (52.51203, 13.3215), (52.51202000000001, 13.321800000000001),
                           (52.51203, 13.322030000000002), (52.512060000000005, 13.322260000000002),
                           (52.512150000000005, 13.322560000000001), (52.512280000000004, 13.32277),
                           (52.512350000000005, 13.322840000000001), (52.51240000000001, 13.322880000000001),
                           (52.51249000000001, 13.323070000000001), (52.512530000000005, 13.32314),
                           (52.512550000000005, 13.32319), (52.512600000000006, 13.32333), (52.51263, 13.32342),
                           (52.51265000000001, 13.323550000000001), (52.512950000000004, 13.32801),
                           (52.513180000000006, 13.33182), (52.513470000000005, 13.33604),
                           (52.5142, 13.346560000000002), (52.51433, 13.348690000000001), (52.51429, 13.34889),
                           (52.51415, 13.349290000000002), (52.51404, 13.349480000000002),
                           (52.513960000000004, 13.349680000000001), (52.51393, 13.349810000000002),
                           (52.51391, 13.350100000000001), (52.51393, 13.35035),
                           (52.513980000000004, 13.350570000000001), (52.514050000000005, 13.350740000000002),
                           (52.514190000000006, 13.350950000000001), (52.51424, 13.350990000000001),
                           (52.51444000000001, 13.351400000000002), (52.51453000000001, 13.351650000000001),
                           (52.5146, 13.352200000000002), (52.51512, 13.36029), (52.51549000000001, 13.36617),
                           (52.51567000000001, 13.369250000000001), (52.515950000000004, 13.37339),
                           (52.51612, 13.376000000000001), (52.51615, 13.376740000000002),
                           (52.51603000000001, 13.37682), (52.51596000000001, 13.376920000000002),
                           (52.51585000000001, 13.37719), (52.51578000000001, 13.37733), (52.515710000000006, 13.37742),
                           (52.515600000000006, 13.37747), (52.515480000000004, 13.37747),
                           (52.51491000000001, 13.37738), (52.51458, 13.377360000000001),
                           (52.514630000000004, 13.378250000000001), (52.514680000000006, 13.379040000000002),
                           (52.51485, 13.379980000000002), (52.515150000000006, 13.381620000000002),
                           (52.51521, 13.3823), (52.515350000000005, 13.38447),
                           (52.515460000000004, 13.386030000000002), (52.51586, 13.38597),
                           (52.51628, 13.385900000000001), (52.51668, 13.385860000000001), (52.51675, 13.38733),
                           (52.51682, 13.388470000000002), (52.51688000000001, 13.3892),
                           (52.51690000000001, 13.389650000000001), (52.51699000000001, 13.39024),
                           (52.517010000000006, 13.3907), (52.51711, 13.392230000000001),
                           (52.51717000000001, 13.392970000000002), (52.51724, 13.39333), (52.51731, 13.39413),
                           (52.517340000000004, 13.394860000000001), (52.517430000000004, 13.39628),
                           (52.517500000000005, 13.397430000000002), (52.51762, 13.398850000000001),
                           (52.517720000000004, 13.39943), (52.517790000000005, 13.39971),
                           (52.517900000000004, 13.400020000000001), (52.51796, 13.400260000000001),
                           (52.51803, 13.400490000000001), (52.518640000000005, 13.4021), (52.51887000000001, 13.40262),
                           (52.519000000000005, 13.40295), (52.51939, 13.4037),
                           (52.519890000000004, 13.404660000000002), (52.520010000000006, 13.404950000000001)]
    now = dt.datetime.now()
    print(get_fill_instructions_for_google_path(path_potsdam_berlin, path_length_km=38.5, start_time=now, speed_kmh=30,
                                                capacity_l=50, start_fuel_l=1))
    print('Bertas Route')
    ROUTE_PATH = os.path.join('..', '..', 'data', 'raw', 'input_data', 'Eingabedaten', 'Fahrzeugrouten')
    BERTA_ROUTE_PATH = os.path.join(ROUTE_PATH, 'Bertha Benz Memorial Route.csv')
    print(get_fill_instructions_for_route(BERTA_ROUTE_PATH))
