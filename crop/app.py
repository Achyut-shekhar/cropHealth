from flask import Flask, request, jsonify, render_template
import rasterio
import numpy as np
import pickle
import os

# Load the pickled model
with open('./model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

# Root route: display upload form
@app.route('/')
def index():
    return render_template('upload.html')

# Handle favicon requests (optional)
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('upload.html', result="⚠️ No file uploaded!")

        file = request.files['file']

        # Check file type
        if not file.filename.endswith('.tif'):
            return render_template('upload.html', result="⚠️ File is not a .tif image!")

        # Read GeoTIFF bands
        with rasterio.open(file) as src:
            ndvi = src.read(1)
            thermal = src.read(2)
            elevation = src.read(3)

        # Compute mean values (ignore NaNs)
        ndvi_mean = np.nanmean(ndvi)
        thermal_mean = np.nanmean(thermal)
        elevation_mean = np.nanmean(elevation)

        # Dummy value for DTM mean
        dtm_mean = 200  # Replace with actual if needed

        # Format input for model
        features_array = np.array([
            ndvi_mean,
            thermal_mean,
            elevation_mean,
            dtm_mean
        ]).reshape(1, -1)

        # Predict
        prediction = model.predict(features_array)
        prediction_class = int(prediction[0][0] > 0.5)

        result = "Healthy 🌱" if prediction_class == 1 else "Unhealthy 🛑"
        return render_template('upload.html', result=result)

    except Exception as e:
        return render_template('upload.html', result=f"❌ Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
