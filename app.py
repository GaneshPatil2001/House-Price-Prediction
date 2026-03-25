import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scalling.pkl', 'rb')) 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST']) 
def predict_api():
    try:
        data = request.json['data']
        
        # Convert values to a list and reshape for the scaler
        input_values = np.array(list(data.values())).reshape(1, -1)
        
        # Scale and Predict
        scaled_data = scalar.transform(input_values)
        prediction = regmodel.predict(scaled_data)

        return jsonify(float(prediction[0]))

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract features in the EXACT order the model expects:
        features = [
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveBedrms']),
            float(request.form['Population']),
            float(request.form['AveOccup']),
            float(request.form['Latitude']),
            float(request.form['Longitude'])
        ]
        
        # Transform the data
        final_input = scalar.transform(np.array(features).reshape(1, -1))
        
        # Predict
        prediction = regmodel.predict(final_input)[0]
        
        formatted_prediction = f"${round(prediction * 100000, 2):,}"

        return render_template("home.html", prediction_text=formatted_prediction)

    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)