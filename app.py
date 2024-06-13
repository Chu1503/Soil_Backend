from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS 

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return 'Soil Backend is running!'

linear_regression_model = joblib.load('linear_regression_model.pkl')
gradient_boosting_model = joblib.load('gradient_boosting_regressor_model.pkl')
neural_network_model = joblib.load('neural_network_model.pkl')

@app.route('/predict_linearregressor', methods=['POST', 'OPTIONS'])
def predict_linear():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        data = request.json
        features = np.array([data['pH'], data['EC'], data['Ava_N'], data['Ava_P'], data['Ava_K']]).reshape(1, -1)
        prediction = linear_regression_model.predict(features)
        response = jsonify({'predicted_OC': prediction[0]})

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-Auth-Token, Origin, Authorization')

    return response

@app.route('/predict_gradientboostingregressor', methods=['POST', 'OPTIONS'])
def predict_gradient_boosting():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        data = request.json
        features = np.array([data['pH'], data['EC'], data['Ava_N'], data['Ava_P'], data['Ava_K']]).reshape(1, -1)
        prediction = gradient_boosting_model.predict(features)
        response = jsonify({'predicted_OC': prediction[0]})

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-Auth-Token, Origin, Authorization')

    return response

@app.route('/predict_neuralnetwork', methods=['POST', 'OPTIONS'])
def predict_neural_network():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        data = request.json
        features = np.array([data['pH'], data['EC'], data['Ava_N'], data['Ava_P'], data['Ava_K']]).reshape(1, -1)
        prediction = neural_network_model.predict(features)
        response = jsonify({'predicted_OC': prediction[0]})

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-Auth-Token, Origin, Authorization')

    return response

if __name__ == '__main__':
    app.run(port=8080, debug=True)