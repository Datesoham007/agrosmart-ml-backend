from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and preprocess data
def load_and_train_model():
    try:
        # Load the dataset
        data = pd.read_csv('soil_crop_recommendation.csv')
        
        # Select features and target
        X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = data['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained successfully with accuracy: {accuracy*100:.2f}%")
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/crop_model.pkl'
        joblib.dump(model, model_path)
        return model_path
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        return None

# Load or train the model
model_path = 'models/crop_model.pkl'
if not os.path.exists(model_path):
    print("Model not found. Training a new model...")
    model_path = load_and_train_model()

try:
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not available", "status": "failed"}), 500
    
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(field in data for field in required_fields):
            return jsonify({
                "error": "Missing required fields",
                "required": required_fields,
                "status": "failed"
            }), 400
        
        # Prepare features
        features = np.array([
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100  # Get the highest probability
        
        return jsonify({
            "recommended_crop": str(prediction),
            "confidence": round(confidence, 2),
            "status": "success"
        })
        
    except ValueError as ve:
        return jsonify({
            "error": f"Invalid input values: {str(ve)}",
            "status": "failed"
        }), 400
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "status": "failed"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
