# 🌾 Crop Health Prediction using Remote Sensing Data

This project uses deep learning and remote sensing data to predict crop health based on NDVI (Normalized Difference Vegetation Index), thermal imagery, and elevation models. It processes GeoTIFF datasets and applies a Convolutional Neural Network (CNN) model for classification of crop health conditions.

## 🛰 Dataset

The dataset is **not included** in this repository due to its large size. You can download the data separately from the official DroneMapper GitHub repository:

**🔗 Download link**: [DroneMapper Crop Analysis Data](https://github.com/dronemapper-io/CropAnalysis)

### 📦 How to use the dataset:

1. Visit the [DroneMapper CropAnalysis GitHub page](https://github.com/dronemapper-io/CropAnalysis).
2. Download the entire repository as a ZIP or clone it using Git:
   ```bash
   git clone https://github.com/dronemapper-io/CropAnalysis.git
Extract or copy the necessary GeoTIFF files from the CropAnalysis directory.

Place all GeoTIFF files into the data/ directory of this project:

kotlin
Copy
Edit
your-project/
├── data/
│   ├── DEM.tif
│   ├── DTM.tif
│   ├── NDVI.tif
│   └── thermal.tif
└── notebooks/
    └── crop_health_model.ipynb
📚 Features
NDVI, thermal, and elevation raster processing

Geospatial analysis with rasterio, geopandas, and rasterstats

Image stacking and normalization for CNN input

TensorFlow model training and evaluation

Health classification visualization using matplotlib

⚙️ Requirements
Install the required dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
Dependencies typically include:

rasterio

geopandas

numpy

matplotlib

tensorflow

scikit-learn

rasterstats

opencv-python

🚀 Run the Project
Ensure your GeoTIFF files are inside the data/ directory.

Open the main Jupyter notebook:

bash
Copy
Edit
jupyter notebook notebooks/crop_health_model.ipynb
Follow the cells in the notebook to:

Load and preprocess the data

Stack raster layers

Train and evaluate the CNN model

Visualize prediction results

📁 Project Structure
bash
Copy
Edit
.
├── data/                     # GeoTIFF files (DEM, NDVI, thermal, etc.)
├── notebooks/
│   └── crop_health_model.ipynb
├── models/                   # Saved model files
├── utils/                    # Helper scripts (e.g., data preprocessing)
├── README.md
└── requirements.txt
🧠 Model
Architecture: CNN built using TensorFlow/Keras.

Input: Multi-band raster created by stacking NDVI, thermal, and elevation data.

Output: Crop health class (e.g., healthy, moderate, stressed).

📝 Notes
Polygon shapefiles can optionally be used to define labeled regions for supervised training.

All raster files are normalized and resized before feeding into the model.

You can expand this project with temporal data, multispectral imagery, or integrate it with web-based dashboards.

📃 License
This project is for educational and research purposes.
Data courtesy of DroneMapper.

Happy Farming! 🌱

yaml
Copy
Edit

---

Let me know if you'd like help generating the `requirements.txt` file or setting up this project 
