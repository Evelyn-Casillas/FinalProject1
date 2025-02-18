import pickle
import numpy as np
from flask import Flask, render_template, request

# Load the trained model (ensure this path is correct)
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Renders the user input form

@app.route('/predict', methods=['POST'])
def predict():
    # Collect only 3 features from the form
    features = [
        float(request.form.get('feature1')),
        float(request.form.get('feature2')),
        float(request.form.get('feature3'))
    ]

    # Convert the features to a numpy array and reshape it for the model
    features = np.array(features).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)

    # Return the prediction to the user
    return render_template('index.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
