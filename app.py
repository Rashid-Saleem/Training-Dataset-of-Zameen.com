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
# NEW: Route for form-based prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        beds = float(request.form['beds'])
        bath = float(request.form['bath'])
        area_num = float(request.form['area_num'])
        sublocation_enc = float(request.form['sublocation_enc'])
        maincity_enc = float(request.form['maincity_enc'])
        is_furnished = float(request.form['is_furnished'])
        is_brand_new = float(request.form['is_brand_new'])
        
        # Create data array
        data = np.array([[beds, bath, area_num, sublocation_enc, 
                         maincity_enc, is_furnished, is_brand_new]])
        
        # Scale and predict
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        
        # Convert to float and format
        predicted_price = float(prediction[0])
        
        return render_template('home.html', 
                              prediction_text=f'Predicted House Price: ${predicted_price:,.2f}',
                              beds=beds, bath=bath, area_num=area_num,
                              sublocation_enc=sublocation_enc, maincity_enc=maincity_enc,
                              is_furnished=is_furnished, is_brand_new=is_brand_new)
    
    except Exception as e:
        return render_template('home.html', 
                              prediction_text=f'Error: {str(e)}')










if __name__=="__main__":
    app.run(debug=True)



