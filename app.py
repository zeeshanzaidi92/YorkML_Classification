from flask import Flask, request, url_for, redirect, render_template, jsonify
pip install pycaret
import numpy as np
import pandas as pd
import pickle
from pycaret.classification import *

app = Flask(__name__)

model = load_model('Final-ML1000-P1-Heart-20211015')
cols = ['Age', 'Gender', 'Chest_Pain', 'Blood_Pressure', 'Cholesterol', 'Blood_Sugar', 'Rest_ECG', 'Max_Heart_Rate','Exercise_Angina', 'ST_Depression', 'ST_Slope', 'Marked_Vessels', 'Thallium']
   
@app.route('/')
def home():
        return render_template('home.html')
   
@app.route('predict', methods = ['POST'])
def predict():
        int_features = [x for x in request.form.values()]
        final = np.array(int_features)
        data_unseen = pd.DataFrame([final], columns = cols)
        prediction = predict_model(model, data = data_unseen, round = 0)
        prediction = int(prediction.Label[0])
        return render_template('home.html', pred = 'Heart Disease {}'.format(prediction))
