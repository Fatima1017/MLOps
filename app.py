from flask import Flask, request, jsonify
import joblib
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, r2_score

# Load the trained model
model = joblib.load('my_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Function to get live data from Open-Meteo API (or another weather API)
def get_live_data():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.52,       # Berlin's latitude
    "longitude": 13.41,      # Berlin's longitude
    "hourly": ",".join([
        'temperature_2m' ,
    'relative_humidity_2m',
    'apparent_temperature',
    'precipitation_probability',
    'rain', 
    'pressure_msl',
    'cloud_cover' ,
    'visibility' ,
    'wind_speed_10m' ,
    'wind_speed_80m' ,
    'wind_speed_120m', 
    'wind_speed_180m' ,
    'temperature_80m' ,
    'temperature_120m' ,
    'temperature_180m' 
    ]),
    "timezone": "auto" }
    response = requests.get(url, params=params)
    weather_data = response.json()

    # Extract data from the response
    live_data = {
    	'longitude': weather_data['longitude'],
    	'latitude': weather_data['latitude'],
    	'timezone': weather_data['timezone'],
        'temperature_2m': weather_data['hourly']['temperature_2m'][0],
        'humidity': weather_data['hourly']['relative_humidity_2m'][0],  # Corrected key
        'wind_speed_10m': weather_data['hourly']['wind_speed_10m'][0],
        'precipitation_probability': weather_data['hourly']['precipitation_probability'][0],
        'rain': weather_data['hourly']['rain'][0],
        'pressure_msl': weather_data['hourly']['pressure_msl'][0],
        'cloud_cover': weather_data['hourly']['cloud_cover'][0],
        'visibility': weather_data['hourly']['visibility'][0],
        'wind_speed_80m': weather_data['hourly']['wind_speed_80m'][0],
        'wind_speed_120m': weather_data['hourly']['wind_speed_120m'][0],
        'wind_speed_180m': weather_data['hourly']['wind_speed_180m'][0],
        'temperature_80m': weather_data['hourly']['temperature_80m'][0],
        'temperature_120m': weather_data['hourly']['temperature_120m'][0], 
        'temperature_180m': weather_data['hourly']['temperature_180m'][0]

    }
    return live_data

# Live dashboard route
@app.route('/')
def live_dashboard():
    live_data = get_live_data()  # Fetch live data from the API
    input_features = np.array(list(live_data.values())).reshape(1, -1)
    #prediction = model.predict(input_features)[0]

    # Render the live data, prediction, and model metrics
    return render_template('dashboard.html', live_data=live_data)

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data:", data)  # Debugging line to check incoming data

    # Ensure 'features' key exists in the data
    if 'features' not in data:
        return jsonify({'error': 'No features provided'}), 400

    # Convert input features to numpy array and reshape
    try:
        input_features = np.array(data['features']).reshape(1, -1)  # Reshape to 2D array
        print("Input features shape:", input_features.shape)  # Debugging line to check the shape

        # Check if the input features have the right number of columns
        if input_features.shape[1] != 14:
            return jsonify({'error': f'Expected 14 features, but got {input_features.shape[1]}'}), 400
        
        # Make the prediction
        prediction = model.predict(input_features)

        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# API endpoint for model accuracy
@app.route('/accuracy', methods=['POST'])
def accuracy():
    # Extract the form data
    form_data = request.form.get((f'feature{i+1}') for i in range(14) ) # Adjust for the number of features
    actual_target = request.form.get('target') # Get the actual target value
    
    input_features = np.array(form_data).reshape(1, -1)  # Reshape input features for the model
    y_pred = model.predict(input_features)[0]  # Get prediction
    
    # Evaluate the model performance
    mse = mean_squared_error([actual_target], [y_pred])
    r2 = r2_score([actual_target], [y_pred])

    # Render the results in an HTML page
    return render_template('accuracy.html', mse=mse, r2=r2)

@app.route('/get_accuracy')
def get_accuracy_form():
    return render_template('accuracy.html')  # Render form for accuracy input


if __name__ == '__main__':
    app.run(debug=True)
