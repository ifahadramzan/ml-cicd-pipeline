import pickle
import os
import math
import numpy as np
from sklearn.linear_model import LinearRegression

class MLModel:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join('models', 'model.pkl')
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Load the model if it exists, otherwise train a new one"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            # Train a simple model with more data points
            X = np.array([[1], [2], [3], [4], [5], [6], [7]])
            y = np.array([2, 4, 6, 8, 10, 12, 14])
            self.model = LinearRegression()
            self.model.fit(X, y)
            # Save the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

    def predict(self, features):
        """Make a prediction based on the input features"""
        features_array = np.array(features).reshape(-1, 1)
        return self.model.predict(features_array).tolist()