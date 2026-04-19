from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    gdp = float(request.form['gdp'])
    freedom = float(request.form['freedom'])
    generosity = float(request.form['generosity'])

    input_data = np.array([[gdp, freedom, generosity]])
    prediction = model.predict(input_data)

    return render_template("index.html", result=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)