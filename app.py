#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
import os
from flask import Flask, session, request, jsonify
import logging
import pickle

app = Flask(__name__)

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)  # Set the desired log level for the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)



def prepare_data(data):
    logger.info("Converting the json input into dataframe")
    # Convert the data to a DataFrame
    new_data = pd.DataFrame([data])

    # Further data preprocessing if required (e.g., converting categorical variables)
    logger.info(f"Data is converted to dataframe: {new_data.head()}")
    return new_data

def load_model():
    logger.info("Loading the model pickle file")
    with open('trained_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def make_predictions(data):
    new_data = prepare_data(data)
    logger.info(f"Data is received in required format {new_data.head(2)}")
    loaded_model = load_model()
    logger.info("Model is loaded successfully")
    predictions = loaded_model.predict(new_data)
    logger.info(f"Predictions received from back {predictions.tolist()}")
    return predictions.tolist()

@app.route('/smartgoal', methods=['post'])
def smartgoal():
    logger.info("Smart Goal Api Invoked successfully")
    data = request.json
    logger.info(f"Data Received as Input: {data}")
    data = request.json  # Assuming the input is received as JSON
    predictions = make_predictions(data)
    logger.info("Predictions recieved successfully")
    return {'predictions': predictions}


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)