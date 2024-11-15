from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the best model and scaler
model = joblib.load("c:\\Users\\NEYHO\\Desktop\\AIML\\Assignment-2\\best_model.sav")

# Initialize the scaler (adjust with the appropriate values if different)
scaler = StandardScaler()
scaler.fit([[0.805, 0.87]])  # Replace with training data stats

# Define the feature columns expected by the model
feature_columns = ["Women's Empowerment Index (WEI) - 2022", "Global Gender Parity Index (GGPI) - 2022"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather input data from the form
        input_features = [float(request.form.get(feature, 0)) for feature in feature_columns]
        input_features = np.array(input_features).reshape(1, -1)
        input_features = scaler.transform(input_features)  # Standardize the input data

        # Predict using the loaded model
        prediction = model.predict(input_features)[0]
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        print(f"Error: {e}")
        return redirect(url_for('index'))  # Redirect to home on error

if __name__ == '__main__':
    app.run(debug=True)
