import os

RAW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
PROCESSED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

GAS_STATIONS_PATH = os.path.join(PROCESSED_PATH, 'Tankstellen_states.csv')
GAS_PRICE_PATH = os.path.join(RAW_PATH, 'input_data', 'Eingabedaten', 'Benzinpreise')
MODEL_PATH = os.path.join(MODELS_PATH, "model_{}.pkl")
