import datetime as dt

from flask import Flask, request, jsonify

import src.serving.route_prediction as prediction

app = Flask(__name__)


@app.route('/prediction/berta', methods=['POST'])
def get_prediction_for_route():
    fuel = request.args.get('fuel', default=0, type=int)
    f = request.files['route']
    result = prediction.get_fill_instructions_for_route(f, start_fuel=fuel)
    print(result)
    return jsonify(result)



@app.route('/prediction/google', methods=['POST'])
def get_prediction():
    required_keys = {'length', 'speed', 'fuel', 'capacity'}
    missing_keys = set(request.json.keys()) - required_keys
    if missing_keys:
        return "Error: also expected this key(s): {}".format(missing_keys)

    val = request.json
    path = [(p[0], p[1]) for p in val['path']]
    length = val['length']
    start_time = dt.datetime.now()
    speed = val['speed']
    fuel = val['fuel']
    capacity = val['capacity']
    result = prediction.get_fill_instructions_for_google_path(path, path_length_km=length, start_time=start_time,
                                                              speed_kmh=speed, capacity_l=capacity, start_fuel_l=fuel)
    return jsonify(result)


if __name__ == "__main__":
    app.run()
