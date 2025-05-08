from flask import Flask, request, render_template  
import joblib 
from joblib import load
import sklearn  
import pickle, gzip  
import pandas as pd  
import numpy as np  
app = Flask(__name__)  

model = load(r"Heart.pkl")

@app.route('/')  
def home():  
  return render_template("home.html")  
@app.route("/predict", methods=["POST"])  
def predict():  
  age = request.form["age"]  
  sex = request.form["sex"]  
  cp = request.form["cp"] 
  trestbps = request.form["trestbps"]  
  chol = request.form["chol"]
  fbs = request.form["fbs"]  
  restecg = request.form["restecg"]    
  thalach = request.form["thalach"]  
  exang = request.form["exang"]  
  oldpeak = request.form["oldpeak"]   
  slope = request.form["slope"]   
  ca = request.form["ca"]
  thal = request.form["thal"]  
  arr = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])  
  pred = model.predict(arr)  
  if pred == 0:  
    res_val = "NO HEART PROBLEM"  
  else:  
    res_val = "HEART PROBLEM"  
  return render_template('home.html', prediction_text='PATIENT HAS {}'.format(res_val))  
if __name__ == "__main__":  
  app.run(debug=True)  
