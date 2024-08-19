from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('trained_models/ssm/dense/model.keras')

def format_to_array(data, range): 
   df = pd.DataFrame(data, columns=['time', 'theta', 'thetadot', 'x', 'xdot'])
   n = len(df.index)-range
   df = df.drop(columns='time')
   df = df.drop(df.index[:n])
   return df

def get_prediction(data):
   df = format_to_array(data, 1)
   df = df.values.reshape(-1, 1, 4)
   history = model.predict(df)
   res = history[0][0].tolist()
   return res

def get_conv_prediction(data):
   df = format_to_array(data, 3)
   df = df.values.reshape(-1, 3, 4)
   history = model.predict(df)
   res = history[0].tolist() 
   return res

def get_lstm_prediction(data):
   df = format_to_array(data, 10)
   df = df.values.reshape(-1, 10, 4)
   history = model.predict(df)
   res = history[0].tolist() 
   return res

app = Flask(__name__)
cors = CORS(app)

@app.route("/receiver", methods=["POST"])
def postME():
   data = request.get_json() 
   res = get_prediction(data)
   r = jsonify(res)
   return r
if __name__ == "__main__":
   app.run(debug=True) 
   
