from flask import request, jsonify
from app import app
from app.model import MLModel

model = MLModel()

@app.route('/')
def index():
    return "ML Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data.get('features', [])
    
    try:
        predictions = model.predict(features)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 400