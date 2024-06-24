from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest Classifier model
filename = 'heart_disease_prediction_rf_model.pkl'  # Change the filename if necessary
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collecting form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])  # Assuming binary encoding: 0 for female, 1 for male
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])  # Assuming binary encoding: 0 for False, 1 for True
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])  # Assuming binary encoding: 0 for No, 1 for Yes
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Preprocessing the data
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Debug prints
        print("Input Data:", data)
        
        # Making prediction
        my_prediction = model.predict(data)
        
        # Debug print
        print("Prediction:", my_prediction)
        
        # Rendering result template with prediction
        return render_template('result.html', prediction=my_prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
