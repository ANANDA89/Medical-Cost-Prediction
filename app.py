from flask import Flask, render_template, request
import pickle
import numpy as np
#import datetime
import pandas as pd

lr_model = pickle.load(open('medical_expense.pkl', 'rb'))  # opening pickle file in read mode

app = Flask(__name__)  # initializing Flask app


@app.route("/",methods=['GET'])
def hello():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        d1 = request.form['Sex']
        if d1 == 'Male':
            d1 = 2
        else:
            d1 = 1
        d2 = request.form['Age']
        d3 = request.form['Children']
        d4 = request.form['Smoker']
        if d4 == 'Yes':
            d4 = 2
        else:
            d4 = 1
        d5=request.form['Region']
        if d5 == 'southwest':
            d5 = 1
        elif d5 == 'southeast':
            d5 = 4
        elif d5 == 'northwest':
            d5 = 2
        elif d5 == 'northeast':
            d5 = 3
        arr = np.array([[d1, d2, d3, d4, d5]])
        pred = lr_model.predict(arr)
        return render_template('index.html', prediction_text='The Health expense is {}'.format(round(pred[0],2)))
    else:
        return render_template('index.html')

app.run(debug=True)
#app.run(host="0.0.0.0")
