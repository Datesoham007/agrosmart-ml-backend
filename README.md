# AgroSmart ML Backend

This is the machine learning backend for the AgroSmart application, providing crop recommendation services using a Random Forest Classifier.

## Setup

1. **Install Python 3.8+** if not already installed.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate sample data** (if you don't have your own dataset):
   ```bash
   python generate_sample_data.py
   ```
   This will create a `soil_crop_recommendation.csv` file with synthetic data.

## Running the Server

Start the Flask development server:
```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

## API Endpoints

### Health Check
- **GET** `/health`
  - Check if the server is running and the model is loaded.

### Get Crop Recommendation
- **POST** `/predict`
  - **Request Body (JSON):**
    ```json
    {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 202.9
    }
    ```
  - **Response:**
    ```json
    {
        "recommended_crop": "rice",
        "confidence": 92.5,
        "status": "success"
    }
    ```

## Environment Variables

- `PORT`: Port to run the server on (default: 5000)
- `MODEL_PATH`: Path to the saved model (default: 'models/crop_model.pkl')

## Deployment

For production, consider using a production WSGI server like Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

## Model Training

The model is automatically trained if no saved model is found. To retrain the model, simply delete the `models/crop_model.pkl` file and restart the server.
