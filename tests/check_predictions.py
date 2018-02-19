if __name__ == '__main__':
    # Check correct price prediction
    price_input_path = 'tests/data/Price_Simple.csv'
    price_input = open(price_input_path, 'r').read().splitlines()[0].split(';')
    price_prediction = open('price_prediction.csv', 'r').read().splitlines()[0].split(';')
    assert price_input == price_prediction[:3]
    ground_truth = 1309
    assert int(price_prediction[-1]) - ground_truth <= 15, "Prediction deviation > 15 deci-cents"

    # Check correct route prediction
    route_input_path = 'tests/data/Route_Bertha_Simple.csv'
    route_input = open(route_input_path, 'r').read().splitlines()[1].split(';')
    _, gas_station_id = route_input
    route_prediction = open('route_prediction.csv', 'r').read().splitlines()[0].split(';')
    gas_station_id2, price, liters = route_prediction
    assert gas_station_id == gas_station_id2
    assert liters == '0'
    ground_truth = 1469
    assert int(price) - ground_truth <= 15, "Prediction deviation > 15 deci-cents"