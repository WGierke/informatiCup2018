import datetime as dt
import json
import logging
import os
import sys

sys.path.insert(0, os.getcwd())
from flask import Flask, request

import src.serving.route_prediction as prediction

app = Flask(__name__)

respond_error = lambda message: {'error': message}
algorithm_error = respond_error('Algorithm failed')


def create_response(message, error=False, status=200):
    if error:
        return app.response_class(
            response=json.dumps(respond_error(message)),
            status=status,
            mimetype='application/json'
        )
    return app.response_class(
        response=respond_error(message),
        status=status,
        mimetype='application/json'
    )


@app.route('/prediction/berta', methods=['POST'])
def get_prediction_for_route():
    if request.json is None:
        return create_response("No JSON has been provided", error=True, status=400)

    required_keys = {'fuel'}
    missing_keys = required_keys - set(request.json.keys())
    if missing_keys:
        return create_response("Also expected this key(s): {}".format(missing_keys), error=True, status=400)

    val = request.json
    fuel = val['fuel']

    if request.file is None:
        return create_response("No FILE containing the route has been uploaded", error=True, status=400)

    f = request.files['route']

    try:
        result = prediction.get_fill_instructions_for_route(f, start_fuel=fuel)
    except Exception as e:
        logging.log(e)
        return create_response(algorithm_error, error=True, status=500)

    return create_response(result)


@app.route('/prediction/google', methods=['POST'])
def get_prediction():
    if not request.json:
        return create_response("No JSON has been provided", error=True, status=400)

    required_keys = {'length', 'speed', 'fuel', 'capacity'}
    missing_keys = required_keys - set(request.json.keys())
    if missing_keys:
        return create_response("Also expected this key(s): {}".format(missing_keys), error=True, status=400)

    val = request.json
    path = [(p[0], p[1]) for p in val['path']]
    length = val['length']
    start_time = dt.datetime.now()
    speed = val['speed']
    fuel = val['fuel']
    capacity = val['capacity']
    try:
        result = prediction.get_fill_instructions_for_google_path(path, path_length_km=length, start_time=start_time,
                                                                  speed_kmh=speed, capacity_l=capacity,
                                                                  start_fuel_l=fuel)
    except Exception as e:
        logging.log(e)
        return create_response(algorithm_error, error=True, status=500)

    return create_response(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
