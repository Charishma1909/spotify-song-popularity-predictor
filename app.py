import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Import CORS for cross-origin requests

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to access backend

# Define paths for model and scaler
MODEL_PATH = 'song_popularity_mlp_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Global variables for model and scaler
model = None
scaler = None

# List of expected features in the exact order the model was trained on
# This is crucial for correct predictions!
# Make sure this list matches the columns of X in your training script.
EXPECTED_FEATURES = [
    'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
    'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
    'time_signature', 'valence'
]

def load_artifacts():
    """
    Loads the trained MLPClassifier model and StandardScaler from disk.
    Exits if files are not found, as they are critical for the app to function.
    """
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please ensure it exists in the same directory as app.py.")
        exit() # Exit if model is not found, as app cannot function without it
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded successfully from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {SCALER_PATH}. Please ensure it exists in the same directory as app.py.")
        exit() # Exit if scaler is not found
    except Exception as e:
        print(f"Error loading scaler: {e}")
        exit()

# Load model and scaler when the Flask app starts
with app.app_context():
    load_artifacts()

@app.route('/')
def index():
    """
    Serves the main HTML page for the prediction interface.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive song features, make a prediction, and return the result.
    Expects a JSON payload with feature values.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    input_features = []

    # Validate and collect features in the correct order
    for feature in EXPECTED_FEATURES:
        value = data.get(feature)
        if value is None:
            return jsonify({"error": f"Missing feature: {feature}"}), 400
        try:
            input_features.append(float(value)) # Convert to float
        except ValueError:
            return jsonify({"error": f"Invalid value for feature {feature}: Must be numeric"}), 400

    # Convert input features to a NumPy array and reshape for the scaler
    # It's important to pass a 2D array (1 sample, N features)
    features_array = np.array([input_features])

    # Scale the input features using the loaded scaler
    try:
        scaled_features = scaler.transform(features_array)
    except Exception as e:
        return jsonify({"error": f"Error during feature scaling: {e}"}), 500

    # Make prediction using the loaded model
    try:
        prediction = model.predict(scaled_features)
        # The model predicts 0 or 1. Map these to 'Not Popular' or 'Popular'.
        result = "Popular" if prediction[0] == 1 else "Not Popular"
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {e}"}), 500

    return jsonify({"prediction": result})

# Run the Flask app
if __name__ == '__main__':
    # Create a 'templates' directory if it doesn't exist
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        print(f"Created directory: {template_dir}")

    # Define the content for index.html
    index_html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Song Popularity Predictor</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 1rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 600px;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: #333;
        }
        input[type="number"] {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #d1d5db;
            border-radius: 0.5rem;
            font-size: 1rem;
            transition: border-color 0.2s;
        }
        input[type="number"]:focus {
            outline: none;
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
        }
        button {
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        button:active {
            transform: scale(0.98);
        }
        .btn-primary {
            background-color: #4f46e5;
            color: white;
            border: none;
        }
        .btn-primary:hover {
            background-color: #4338ca;
        }
        .btn-secondary {
            background-color: #e5e7eb;
            color: #374151;
            border: 1px solid #d1d5db;
        }
        .btn-secondary:hover {
            background-color: #d1d5db;
        }
        #result {
            margin-top: 1.5rem;
            padding: 1rem;
            border-radius: 0.5rem;
            font-size: 1.125rem;
            font-weight: 700;
            text-align: center;
            background-color: #eff6ff;
            color: #1e40af;
            border: 1px solid #bfdbfe;
            display: none; /* Hidden by default */
        }
        .loading-spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left-color: #4f46e5;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 8px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="container">
        <h1 class="text-3xl font-extrabold text-center text-gray-800 mb-8">
            🎶 Song Popularity Predictor 🎶
        </h1>

        <form id="predictionForm" class="space-y-4">
            <!-- Input fields for each feature -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="form-group">
                    <label for="acousticness">Acousticness:</label>
                    <input type="number" id="acousticness" step="0.01" min="0" max="1" value="0.2" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="danceability">Danceability:</label>
                    <input type="number" id="danceability" step="0.01" min="0" max="1" value="0.7" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="duration_ms">Duration (ms):</label>
                    <input type="number" id="duration_ms" step="1" min="0" value="200000" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="energy">Energy:</label>
                    <input type="number" id="energy" step="0.01" min="0" max="1" value="0.8" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="instrumentalness">Instrumentalness:</label>
                    <input type="number" id="instrumentalness" step="0.01" min="0" max="1" value="0.0" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="key">Key (0-11):</label>
                    <input type="number" id="key" step="1" min="0" max="11" value="5" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="liveness">Liveness:</label>
                    <input type="number" id="liveness" step="0.01" min="0" max="1" value="0.15" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="loudness">Loudness (dB):</label>
                    <input type="number" id="loudness" step="0.01" value="-5.0" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="mode">Mode (0=minor, 1=major):</label>
                    <input type="number" id="mode" step="1" min="0" max="1" value="1" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="speechiness">Speechiness:</label>
                    <input type="number" id="speechiness" step="0.01" min="0" max="1" value="0.05" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="tempo">Tempo (BPM):</label>
                    <input type="number" id="tempo" step="0.01" value="120.0" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="time_signature">Time Signature (3-5):</label>
                    <input type="number" id="time_signature" step="1" min="3" max="5" value="4" required class="rounded-md">
                </div>
                <div class="form-group">
                    <label for="valence">Valence:</label>
                    <input type="number" id="valence" step="0.01" min="0" max="1" value="0.6" required class="rounded-md">
                </div>
            </div>

            <div class="flex justify-center space-x-4 mt-6">
                <button type="submit" class="btn-primary">
                    <span id="predictButtonText">Predict Popularity</span>
                    <span id="loadingSpinner" class="loading-spinner hidden"></span>
                </button>
                <button type="button" id="clearButton" class="btn-secondary">Clear</button>
            </div>
        </form>

        <div id="result" class="mt-6 p-4 rounded-lg text-center font-bold text-lg hidden">
            <!-- Prediction result will be displayed here -->
        </div>
    </div>

    <script>
        const form = document.getElementById('predictionForm');
        const resultDiv = document.getElementById('result');
        const predictButtonText = document.getElementById('predictButtonText');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const clearButton = document.getElementById('clearButton');

        // List of feature IDs to easily collect values
        const featureIds = [
            'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
            'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
            'time_signature', 'valence'
        ];

        form.addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent default form submission

            resultDiv.style.display = 'none'; // Hide previous result
            predictButtonText.textContent = 'Predicting...';
            loadingSpinner.classList.remove('hidden');
            form.querySelector('button[type="submit"]').disabled = true; // Disable button during prediction

            const features = {};
            featureIds.forEach(id => {
                features[id] = document.getElementById(id).value;
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(features)
                });

                const data = await response.json();

                if (response.ok) {
                    resultDiv.textContent = `Prediction: This song is likely ${data.prediction}.`;
                    resultDiv.className = 'mt-6 p-4 rounded-lg text-center font-bold text-lg'; // Reset classes
                    if (data.prediction === 'Popular') {
                        resultDiv.classList.add('bg-green-100', 'text-green-800', 'border-green-300');
                    } else {
                        resultDiv.classList.add('bg-red-100', 'text-red-800', 'border-red-300');
                    }
                } else {
                    resultDiv.textContent = `Error: ${data.error || 'Something went wrong'}`;
                    resultDiv.classList.add('bg-red-100', 'text-red-800', 'border-red-300');
                }
            } catch (error) {
                console.error('Fetch error:', error);
                resultDiv.textContent = 'An error occurred while connecting to the server.';
                resultDiv.classList.add('bg-red-100', 'text-red-800', 'border-red-300');
            } finally {
                resultDiv.style.display = 'block'; // Show result
                predictButtonText.textContent = 'Predict Popularity';
                loadingSpinner.classList.add('hidden');
                form.querySelector('button[type="submit"]').disabled = false; // Re-enable button
            }
        });

        clearButton.addEventListener('click', () => {
            form.reset(); // Resets all form fields to their initial values
            resultDiv.style.display = 'none'; // Hide the result div
            resultDiv.textContent = ''; // Clear result text
            resultDiv.className = 'mt-6 p-4 rounded-lg text-center font-bold text-lg hidden'; // Reset classes
        });
    </script>
</body>
</html>
"""
    # Write the index.html content to the templates directory with UTF-8 encoding
    with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html_content)
    print("index.html created in 'templates' directory.")

    # Run the Flask app in debug mode for development
    app.run(debug=True)