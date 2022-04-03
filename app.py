# import libraries
from flask import Flask, render_template,request,jsonify
from neuralprophet import NeuralProphet, set_log_level
import pandas as pd
from PIL import Image
import base64
import io
import pickle
import os

# remove the previous instance of the forecast.jpg
#os.remove("./forecast.jpg")
# load model the neuralprophet model
model = pickle.load(open('model.pkl','rb'))

# app name
app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# routes
@app.route("/predict", methods=["POST"])

def predict():
    
    # create the pred_gen2 from scratch using plant_1_generation_data.csv
    import pandas as pd
    pred_gen2=pd.read_csv('Plant_1_Generation_Data.csv')
    pred_gen2.drop('PLANT_ID',1,inplace=True)

    #format datetime
    pred_gen2['DATE_TIME']= pd.to_datetime(pred_gen2['DATE_TIME'],format='%d-%m-%Y %H:%M')

    # sum up all the daily yield from all 22 inverters and date time and reset the index
    pred_gen2=pred_gen2.groupby('DATE_TIME')['DAILY_YIELD'].sum().reset_index()
    pred_gen2.rename(columns={'DATE_TIME':'ds','DAILY_YIELD':'y'},inplace=True)
    
    # get periods data from user
    #data = request.get_json(force=True)
    
    periods =  int(request.form['experience_ddn'])
  
    # make predictions
    # periods = 192 corresponds to 2 day ahead forecast
    future = model.make_future_dataframe(pred_gen2,periods =periods ,n_historic_predictions=True) 
    
    forecast = model.predict(future)
    plot1 =model.plot(forecast)
    
    plot1.savefig('./forecast.jpg')
    
    # Full Script.
    im = Image.open("forecast.jpg")
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template("index.html", img_data=encoded_img_data.decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True)

