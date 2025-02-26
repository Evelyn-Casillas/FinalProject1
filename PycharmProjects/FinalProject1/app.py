from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

filename = 'file_model.pkl'
model = pickle.load(open(filename, 'rb'))  # Load the model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    Age = float(request.form['age'])
    Sleep_Duration = float(request.form['sleep_duration'])
    Quality_of_Sleep = float(request.form['quality_of_sleep'])
    Occupation_Salesperson = int(request.form['occupation_salesperson'])
    BMICategory_Overweight = int(request.form['category_overweight'])
    Stress_Level = float(request.form['stress_level'])
    Occupation_Doctor = int(request.form['occupation_doctor'])

    features = np.array([[Age, Sleep_Duration, Quality_of_Sleep, Occupation_Salesperson,
                          BMICategory_Overweight, Stress_Level, Occupation_Doctor]])

    pred = model.predict(features)

    return render_template('index.html',
                           predict=str(pred[0]))  # Assuming the output is a binary classification (0 or 1)


if __name__ == '__main__':
    app.run(debug=True)
