from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import os

app = Flask(__name__)
cors = CORS(app)

@app.route("/receiver", methods=["POST"])
def postME():
   data = request.get_json()
   data = jsonify(data)
   return data
if __name__ == "__main__": 
   app.run(debug=True)