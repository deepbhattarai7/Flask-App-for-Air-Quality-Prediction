from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('app/air_quality_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        no2 = float(request.form['no2'])
        so2 = float(request.form['so2'])
        co = float(request.form['co'])
        proximity = float(request.form['proximity'])
        population = float(request.form['population'])

        # Create DataFrame with proper feature names
        feature_names = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
        features = [[temperature, humidity, pm25, pm10, no2, so2, co, proximity, population]]
        input_df = pd.DataFrame(features, columns=feature_names)

        # Predict
        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f'{prediction}')
    except Exception as e:
        return render_template('index.html', error_message=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
