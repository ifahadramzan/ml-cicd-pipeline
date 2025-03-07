import unittest
from app.model import MLModel

class TestMLModel(unittest.TestCase):
    def setUp(self):
        self.model = MLModel()

    def test_predict(self):
        # Test that prediction works for sample data
        features = [6]
        predictions = self.model.predict(features)
        self.assertEqual(len(predictions), 1)
        # Linear model should predict close to 12 for input 6
        self.assertAlmostEqual(predictions[0], 12, delta=1)

if __name__ == '__main__':
    unittest.main()