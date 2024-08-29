import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open("xgbmodel.pkl", 'rb'))
# scale = pickle.load(open(,'rb'))

@app.route('/')  # route to display the home page
def home():
    return render_template('index.html')  # rendering the home page

@app.route('/predict', methods=["POST", "GET"])  # route to show the predictions in a web UI
def predict():
    # reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = [['Sex', 'Marital status', 'Age', 'Education', 'Income', 'Occupation', 'Settlement size']]
    data = pandas.DataFrame(features_values, columns=names)
    # data = scale.fit_transform(features_values)

    # predictions using the loaded model file
    prediction = model.predict(data)
    print(prediction)

    if prediction == 0:
        return render_template("index.html", prediction_text="Not a potential customer")
    elif prediction == 1:
        return render_template("index.html", prediction_text="Potential customer")
    else:
        return render_template("index.html", prediction_text="Highly potential customer")

# showing the prediction results in a UI
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
