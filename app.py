from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import warnings

# Suppress warnings related to feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load the model
with open('flight_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_input(data):
    # Example mapping - adjust according to your model's feature requirements
    airline_map = {'IndiGo': 0, 'Air India': 1, 'Jet Airways': 2, 'SpiceJet': 3, 'Multiple carriers': 4, 'GoAir': 5}
    source_map = {'Banglore': 0, 'Kolkata': 1, 'Delhi': 2, 'Chennai': 3, 'Mumbai': 4}
    destination_map = {'New Delhi': 0, 'Banglore': 1, 'Cochin': 2, 'Kolkata': 3, 'Delhi': 4, 'Hyderabad': 5}
    additional_info_map = {'No info': 0, 'In-flight meal not included': 1, 'No check-in baggage included': 2}

    # Encode the categorical variables
    airline = airline_map.get(data['airline'], -1)
    source = source_map.get(data['source'], -1)
    destination = destination_map.get(data['destination'], -1)
    additional_info = additional_info_map.get(data['additional_info'], -1)

    # Extract numerical features and ensure they are in the right format
    stops = int(data['stops'])
    journey_day = int(data['journey_day'])
    journey_month = int(data['journey_month'])
    dep_hour = int(data['dep_hour'])
    dep_min = int(data['dep_min'])
    arrival_hour = int(data['arrival_hour'])
    arrival_min = int(data['arrival_min'])

    # Combine all features into a single array
    features = np.array([
        airline, source, destination, stops, journey_day, journey_month,
        dep_hour, dep_min, arrival_hour, arrival_min, additional_info
    ])

    return features.reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from request
        data = request.get_json(force=True)
        
        # Debugging: print the received data
        print("Received input data:", data)

        # Preprocess the data to match the model's expected input format
        input_data = preprocess_input(data)

        # Debugging: print the processed data
        print("Processed input data:", input_data)

        # Make prediction
        prediction = model.predict(input_data)

        # Debugging: print the prediction
        print("Model prediction:", prediction)

        # Send back the result as JSON
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        # Debugging: print the error
        print("Error during prediction:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
