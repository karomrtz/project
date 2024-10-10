import os
import pandas as pd
import pickle
import sys
from logging.handlers import RotatingFileHandler
import logging
from app.helpers.load_data_helpers import  load_parameters,load_data, load_model, load_scaler, save_predictions
from app.helpers.preprocesing_helpers import clean_data, create_feature_data, stantarize_data
import logging


APP_CONFIG = 'app/config/preprocessig_paramas.json'
APP_MODEL = 'app/model/trained_model1.pkl'
APP_SCALER= 'app/model/scaler.pkl'
LOG_FILE = 'app/logs/app.log'
DATA_PATH = 'data/data.csv'
PREDICTIONS_PATH = '/project/data/predictions/predictions.csv'


def main():
    # Configure logs
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_handler = RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  
    log_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s'))
    logger.addHandler(log_handler)
    logger.info('Data process started')
    
    try:
        # Load parameters from JSON file
        params = load_parameters(APP_CONFIG)
        logger.info(f"Parameters loaded Sussefully")
        
        drop_nan_columns = params['drop_nan_columns']
        columns_replace_nan = params['columns_replace_nan']
        binary_columns = params['binary_columns']
        columns_to_onehot = params['columns_to_onehot']
        columns_to_scale = params['columns_to_scale']
        column_types = params['column_types']
        
    except Exception as e:
        logger.exception(f"Error loading parameters: {e}")
        sys.exit(1)
    try:
        # Load trained model
        model = load_model(APP_MODEL)
        logger.info(f"Model loaded successfully")
        # Load scaler
        scaler = load_scaler(APP_SCALER)
        logger.info(f"Scaler loaded successfully")
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        sys.exit(1)
        
    try:
        # Load  data 
        df = load_data(DATA_PATH)
        logger.info(f"Data loaded successfully")

        # Preprocess the data
        df_cleaned = clean_data(df, drop_nan_columns, columns_replace_nan, binary_columns)
        df_with_features = create_feature_data(df_cleaned, columns_to_onehot, scaler, columns_to_scale)
        df_standardized = stantarize_data(df_with_features, column_types)
        logger.info(f"Data preprocessed successfully")
    except Exception as e:
        logger.exception(f"Error preprocessing data: {e}")
        sys.exit(1)
    
    try:
        # Make predictions and save
        predictions = model.predict(df_standardized)
        predictions = pd.DataFrame(predictions, columns=['prediction'])
        logger.info(f"Predictions made successfully")
        logger.info(f"Predictions: {predictions.head()}")
        save_predictions(predictions, PREDICTIONS_PATH)
    except Exception as e:
        logger.exception(f"Error making predictions: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()