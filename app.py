from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Path to the models directory
models_dir = os.path.join(os.getcwd(), 'Models')

# Load models
models = {
    'LinearRegression': joblib.load(os.path.join(models_dir, 'linear_regression_model.pkl')),
    'GradientBoostingRegressor': joblib.load(os.path.join(models_dir, 'gradient_boosting_regressor_model.pkl')),
    # Add more models as needed
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