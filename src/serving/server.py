import datetime as dt
import json
import logging
import os
import sys
import traceback

sys.path.insert(0, os.getcwd())
from flask import Flask, request, send_from_directory

import src.serving.route_prediction as prediction

app = Flask(__name__, static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = float("inf")
DOCS_DIR = os.path.abspath(os.path.join(os.getcwd(), 'docs'))

respond_error = lambda message: {'error': message}
algorithm_error = respond_error('Algorithm failed')


def create_response(message, error=False, status=200):
    print(json.dumps(message))
    if not error:
        return app.response_class(
            response=json.dumps(message),
            status=status,
            mimetype='application/json'
        )
    return app.response_class(
        response=respond_error(message),
        status=status,
        mimetype='application/json'
    )


@app.route('/', methods=['GET'])
def root():
    return send_from_directory(DOCS_DIR, 'index.html')

@app.route('/static/<path:path>', methods=['GET'])
def serve_file_in_dir(path):
    if not os.path.isfile(os.path.join(DOCS_DIR, path)):
        path = os.path.join(path, 'index.html')
    return send_from_directory(DOCS_DIR, path)

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
        result = prediction.get_fill_instructions_for_route(f, start_fuel=fuel, in_euros=True)
    except Exception as e:
        logging.error(e)
        print(val)
        traceback.print_exc()
        return create_response(algorithm_error, error=True, status=500)

    return create_response(result)


@app.route('/prediction/google', methods=['POST'])
def get_prediction():
    val = request.get_json()
    if not val:
        return create_response("No JSON has been provided", error=True, status=400)

    required_keys = {'length', 'speed', 'fuel', 'capacity', 'path'}
    missing_keys = required_keys - set(request.json.keys())
    if missing_keys:
        return create_response("Also expected this key(s): {}".format(missing_keys), error=True, status=400)

    path = [(p[0], p[1]) for p in val['path']]
    length = float(val['length'])  # TODO: read starttime
    start_time = dt.datetime.now()
    speed = float(val['speed'])
    fuel = float(val['fuel'])
    capacity = float(val['capacity'])
    try:
        result = prediction.get_fill_instructions_for_google_path(path, path_length_km=length, start_time=start_time,
                                                                  speed_kmh=speed, capacity_l=capacity,
                                                                  start_fuel_l=fuel)
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        print(val)
        return create_response(algorithm_error, error=True, status=500)

    print(result)
    response = create_response(result)
    print(response)
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0')
