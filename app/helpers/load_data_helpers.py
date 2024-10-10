
import json
import sys
import pandas as pd
import logging
import pickle
import joblib
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_parameters(json_file):
    """
    Load parameters from a JSON file.

    Parameters:
    ----------
    json_file : str
        Path to the JSON file containing parameters.

    Returns:
    -------
    dict
        Dictionary with parameters loaded from the JSON file.
    """
    with open(json_file, 'r') as file:
        parameters = json.load(file)
    if not parameters:
        print(f'Parameters file is empty (at load_parameters)')
        # sys.exit(1)

    return parameters

def load_data(file, **options):
    """
    Loads file.
    
    Parameters
    ----------
    file: Str.
    Name of the dataset to load.

    """  
     
    data = pd.read_csv(file, **options) 
        

        
    if data.empty:
        # print(f'Input table is empty (at load_file)')  
        logger.error(f'Input table is empty (at load_file)')
        sys.exit(1)    
    elif data.isna().all().all():
        logger.error(f'Input table has all NaNs (at load_file)')
        # print(f'Input table has all NaNs (at load_file)')

        # sys.exit(1)
        
    return data

def load_model(file):
    """
    Load a trained model from a pickle file.

    Parameters:
    ----------
    file : str
        Path to the pickle file containing the trained model.

    Returns:
    -------
    object
        The trained model loaded from the pickle file.
    """
    
    with open(file, 'rb') as file:
        model = pickle.load(file)
    return model

def load_scaler(file):
    """
    Load a trained StandardScaler object from a pickle file.

    Parameters:
    ----------
    file : str
        Path to the pickle file containing the trained StandardScaler object.

    Returns:
    -------
    StandardScaler
        The trained StandardScaler object loaded from the pickle file.
    """
    scaler = joblib.load(file)
    return scaler

def save_predictions(predictions, output_file):
    """
    Save model predictions to a CSV file.

    Parameters:
    ----------
    predictions : pd.Series, pd.DataFrame, or list/array-like
        The predictions to save.
    output_file : str
        Path to the output CSV file.
    """
    if not isinstance(predictions, (pd.Series, pd.DataFrame)):
        predictions = pd.DataFrame(predictions, columns=['Predictions'])

    predictions.to_csv(output_file, index=False)

    logger.info(f"Predictions saved to {output_file}")
    
    return None