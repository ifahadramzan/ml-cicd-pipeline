import unittest
import numpy as np
from sklearn.linear_model import LinearRegression

class TestMLModel(unittest.TestCase):
    def test_model_prediction(self):
        # Create a simple model directly for testing
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        model = LinearRegression()
        model.fit(X, y)
        
        # Test prediction
        test_input = np.array([[6]])
        prediction = model.predict(test_input)[0]
        self.assertAlmostEqual(prediction, 12, delta=1)

if __name__ == '__main__':
    unittest.main()