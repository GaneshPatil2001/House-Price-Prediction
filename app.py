import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scalling.pkl', 'rb'))  

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST']) 
def predict_api():
    try:
        data = request.json['data']
        
        # Convert input to numpy array
        input_data = np.array(list(data.values())).reshape(1, -1)
        print("Input Data:", input_data)

        # Scale the data
        new_data = scalar.transform(input_data)  

        # Prediction
        output = regmodel.predict(new_data)
        print("Prediction:", output[0])

        return jsonify({'prediction': float(output[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)