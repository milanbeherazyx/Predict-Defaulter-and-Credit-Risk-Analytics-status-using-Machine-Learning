from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

scaler=pickle.load(open("/config/workspace/models/scaler.pkl", "rb"))
model = pickle.load(open("/config/workspace/models/classifier.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        ed=int(request.form.get("ed"))
        address = int(request.form.get('address'))
        creddebt = float(request.form.get('creddebt'))
        
        new_data=scaler.transform([[ed,address,creddebt]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Never defaulted'
        else:
            result ='defaulted'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")