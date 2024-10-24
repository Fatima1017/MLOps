import unittest
from flask import Flask
from app import app.py  # Import your Flask app. Change 'app' to your file name if different.
import json

class FlaskWeatherAppTestCase(unittest.TestCase):

    def setUp(self):
        """Set up the test client and any necessary test data."""
        self.app = app.test_client()
        self.app.testing = True

    def test_live_dashboard(self):
        """Test the live dashboard route."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'temperature_2m', response.data)  # Update this based on your dashboard template content

    def test_predict(self):
        """Test the prediction API endpoint."""
        # Sample feature values for prediction (update with correct number of features as needed)
        features = [20, 60, 5, 0, 0, 1000, 10, 5, 20, 100, 10, 20, 50, 15]  # Adjust the number and values of features
        response = self.app.post('/predict', json={'features': features})
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('prediction', json_data)

    def test_accuracy(self):
        """Test the accuracy route for model evaluation."""
        # Sample form data for accuracy calculation
        form_data = {
            'feature1': 20,
            'feature2': 60,
            'feature3': 5,
            'feature4': 0,
            'feature5': 0,
            'feature6': 1000,
            'feature7': 10,
            'feature8': 5,
            'feature9': 20,
            'feature10': 100,
            'feature11': 10,
            'feature12': 20,
            'feature13': 50,
            'feature14': 15,  # Example of an actual target value
            'target': 15  # Replace with the actual target value for testing
        }
        response = self.app.post('/accuracy', data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Mean Squared Error', response.data)  # Update based on your HTML output

    def test_get_accuracy_form(self):
        """Test the route that serves the accuracy input form."""
        response = self.app.get('/get_accuracy')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Accuracy', response.data)  # Update based on your HTML content

if __name__ == '__main__':
    unittest.main()
 
