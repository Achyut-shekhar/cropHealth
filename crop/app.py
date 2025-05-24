from flask import Flask, request, render_template
import rasterio
import numpy as np
import traceback
import pickle

app = Flask(__name__)  # Make sure this is defined BEFORE route decorators

# Load the ML model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def safe_read_band(src, band_index):
    try:
        return src.read(band_index)
    except Exception as e:
        print(f"Warning: Could not read band {band_index}. Error: {e}")
        return np.zeros(src.shape, dtype=np.float32)

def calculate_ndvi(nir, red):
    nir = np.abs(np.array(nir, dtype=np.float32))
    red = np.abs(np.array(red, dtype=np.float32))
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red + 1e-10)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    return ndvi

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('upload.html', result="⚠️ No file uploaded!")

        file = request.files['file']
        if not file.filename.endswith('.tif'):
            return render_template('upload.html', result="⚠️ File is not a .tif image!")

        file.seek(0)
        with rasterio.open(file) as src:
            if src.count < 5:
                return render_template('upload.html',
                    result="❌ The image must contain at least 5 bands: Red, NIR, Thermal, Elevation, and DTM.")

            print(f"Total bands: {src.count}, Shape: {src.shape}, Dtype: {src.dtypes}")

            # Read bands
            red       = safe_read_band(src, 1)
            nir       = safe_read_band(src, 2)
            thermal   = safe_read_band(src, 3)
            elevation = safe_read_band(src, 4)
            dtm       = safe_read_band(src, 5)

            # Compute NDVI
            ndvi = calculate_ndvi(nir, red)

            # Mean values
            ndvi_mean = np.nanmean(ndvi)
            thermal_mean = np.nanmean(thermal)
            elevation_mean = np.nanmean(elevation)
            dtm_mean = np.nanmean(dtm)

            # Print mean values for debugging
            print(f"NDVI mean: {ndvi_mean:.4f}")
            print(f"Thermal mean: {thermal_mean:.4f}")
            print(f"Elevation mean: {elevation_mean:.4f}")
            print(f"DTM mean: {dtm_mean:.4f}")

            # Handle NaNs
            if any(np.isnan(v) for v in [ndvi_mean, thermal_mean, elevation_mean, dtm_mean]):
                return render_template('upload.html',
                    result="❌ One or more feature values are invalid (NaN).")

            # Threshold rule: predict 'unhealthy' if NDVI mean below 0.47
            if ndvi_mean < 0.4:
                prediction = 'unhealthy'
                result = f" {prediction}"
            else:
                features = np.array([[ndvi_mean, thermal_mean, elevation_mean, dtm_mean]])
                print("Input features:", features)

                prediction_raw = model.predict(features)
                prediction = prediction_raw[0] if hasattr(prediction_raw, '__getitem__') else prediction_raw

                confidence = f"{min(abs(ndvi_mean - 0.3) / 0.5 * 100, 100):.1f}%"
                result = f"✅ Prediction: Healthy "

            return render_template('upload.html',
                                   result=result,
                                   prediction=prediction,
                        
                                   ndvi=f"{ndvi_mean:.2f}",
                                   thermal=f"{thermal_mean:.2f}",
                                   elevation=f"{elevation_mean:.2f}",
                                   dtm=f"{dtm_mean:.2f}")

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full Error Traceback: {error_details}")
        return render_template('upload.html', result=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
