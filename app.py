from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

linear_regression_model = joblib.load('linear_regression_model.pkl')
gradient_boosting_model = joblib.load('gradient_boosting_regressor_model.pkl')

models = {
    'LinearRegression': linear_regression_model,
    'GradientBoostingRegressor': gradient_boosting_model,
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data.get('model_name')
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 400

    model = models[model_name]
    features = np.array([data['pH'], data['EC'], data['Ava_N'], data['Ava_P'], data['Ava_K']]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'predicted_OC': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
