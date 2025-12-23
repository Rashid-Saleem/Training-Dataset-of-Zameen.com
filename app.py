import pickle
from flask import Flask,jsonify,app,url_for,render_template,request
import numpy as np
import pandas as pd

app=Flask(__name__)

##Load the model

model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)#This will get data in key value pair of Dictionary
    #Now m going to convert this data into List 
    print(np.array(list(data.values())).reshape(1,-1))
    newData=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(newData)
    prediction=float(output[0])
    return jsonify(prediction)

if __name__=="__main__":
    app.run(debug=True)



