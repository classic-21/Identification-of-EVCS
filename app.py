from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
with open('model_and_scaler.pkl', 'rb') as f:
    saved_objects = pickle.load(f)
    model = saved_objects['model']
    scaler = saved_objects['scaler']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        features = [float(request.form[f'feature{i+1}']) for i in range(8)]
        input_data = np.array([features])

        # Scale input data
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)[0]

        # Check condition with Feature 7
        feature_7_value = features[4]  # Feature 7 is at index 6
        if prediction > feature_7_value:
            difference = prediction - feature_7_value
            message = f"We need {difference:.0f} more EV charging stations here."
            # Return result
            return render_template(
                'index.html',
                prediction_text=f'Total required stations are {prediction:.0f}',
                condition_message=message
            )
        else:
            message = "There are sufficient stations."

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"An error occurred: {e}"
        )

if __name__ == '__main__':
    app.run(debug=True)
