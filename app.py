from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, Predict_Pipeline

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['GET' , 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        predict_pipeline = Predict_Pipeline()
        data = CustomData(
            gender=request.form['gender'],
            race=request.form['ethnicity'],
            parental_edu=request.form['parental_level_of_education'],
            lunch=request.form['lunch'],
            test_prep_course=request.form['test_preparation_course'],
            reading_score=request.form['reading_score'],
            writing_score=request.form['writing_score'],
        )
        features = data.get_data_as_DataFrame()
        prediction = predict_pipeline.predict(features)
        return render_template('home.html', results=prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)