from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('final_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        cgpa = float(request.form['cgpa'])
        iq = float(request.form['iq'])

        # Make prediction
        features = np.array([[cgpa, iq]])
        prediction = model.predict(features)[0]

        # Translate result
        result = "✅ Likely to be placed!" if prediction == 1 else "❌ May not be placed."

        return render_template('index.html', prediction=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
