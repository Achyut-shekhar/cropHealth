from flask import Flask, request, render_template
import rasterio
import numpy as np
import traceback
import pickle

app = Flask(__name__)

# Load the ML model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def safe_read_band(src, band_index):
    try:
        return src.read(band_index)
    except Exception as e:
        print(f"Warning: Could not read band {band_index}. Error: {e}")
        return np.zeros(src.shape, dtype=np.float32)

def calculate_ndvi(near_infrared, red):
    near_infrared = np.abs(np.array(near_infrared, dtype=np.float32))
    red = np.abs(np.array(red, dtype=np.float32))
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (near_infrared - red) / (near_infrared + red + 1e-10)
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
            num_bands = src.count
            print(f"Total bands: {num_bands}, Shape: {src.shape}, Dtype: {src.dtypes}")

            band_details = []
            for i in range(1, num_bands + 1):
                try:
                    band = src.read(i)
                    band_details.append({
                        'index': i,
                        'min': np.nanmin(band),
                        'max': np.nanmax(band),
                        'mean': np.nanmean(band),
                        'dtype': band.dtype
                    })
                except Exception as e:
                    print(f"Error reading band {i}: {e}")

            possible_combinations = []
            for red_idx in range(1, num_bands + 1):
                for nir_idx in range(1, num_bands + 1):
                    if red_idx == nir_idx:
                        continue
                    try:
                        red = safe_read_band(src, red_idx)
                        nir = safe_read_band(src, nir_idx)
                        ndvi = calculate_ndvi(nir, red)
                        ndvi_mean = np.nanmean(ndvi)
                        if not np.isnan(ndvi_mean):
                            possible_combinations.append({
                                'red_band': red_idx,
                                'nir_band': nir_idx,
                                'ndvi_mean': ndvi_mean,
                                'ndvi_min': np.nanmin(ndvi),
                                'ndvi_max': np.nanmax(ndvi)
                            })
                    except Exception as e:
                        print(f"Error processing bands {red_idx}-{nir_idx}: {e}")

            if not possible_combinations:
                band_info = ", ".join([f"Band {d['index']}: [{d['min']}, {d['max']}]" for d in band_details])
                return render_template('upload.html',
                                       result=f"❌ Could not calculate NDVI. Band Details: {band_info}")

            best_combo = max(possible_combinations, key=lambda x: x['ndvi_mean'])
            ndvi_value = best_combo['ndvi_mean']
            ndvi_threshold = 0.3

            if ndvi_value < ndvi_threshold:
                status = "Unhealthy 🛑"
                details = f"NDVI ({ndvi_value:.2f}) is below threshold ({ndvi_threshold})"
                prediction = "N/A"
                confidence = "N/A"
            else:
                status = "Healthy 🌱"
                details = f"NDVI ({ndvi_value:.2f}) is above threshold ({ndvi_threshold})"
                try:
                    prediction_raw = model.predict([[ndvi_value]])
                    print("Prediction raw:", prediction_raw, type(prediction_raw))
                    if hasattr(prediction_raw, '__getitem__'):
                        prediction = prediction_raw[0]
                    else:
                        prediction = prediction_raw
                    confidence_value = abs(ndvi_value - ndvi_threshold) / 0.5
                    confidence = f"{min(confidence_value * 100, 100):.1f}%"
                except Exception as e:
                    prediction = f"⚠️ Prediction error: {e}"
                    confidence = "N/A"

            result = f"{status}\n{details}"
            return render_template('upload.html',
                                   result=result,
                                   confidence=confidence,
                                   ndvi=f"{ndvi_value:.2f}",
                                   ndvi_min=f"{best_combo['ndvi_min']:.2f}",
                                   ndvi_max=f"{best_combo['ndvi_max']:.2f}",
                                   combo=f"Bands {best_combo['red_band']}-{best_combo['nir_band']}",
                                   prediction=prediction)

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full Error Traceback: {error_details}")
        return render_template('upload.html', result=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
