
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_missing_data(df,drop_nan_columns, columns_replace_nan):
    """
    Handle missing data by dropping columns with NaN values and replacing NaN values in specified columns.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame with missing data.
    drop_nan_columns : list
        List of columns where rows with NaN should be dropped.
    columns_replace_nan : dict
        Dictionary where keys are columns and values are the replacement values for NaN.

    Returns:
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """

    if drop_nan_columns:
        df.dropna(subset=drop_nan_columns, inplace=True)
    
    if columns_replace_nan:
        for column, value in columns_replace_nan.items():
            if column in df.columns:
                df[column] = df[column].fillna(value)
    
    return df

def replace_binary_values(dataframe, binary_columns):
    """
    Replace 'Yes'/'No' binary values with 1 and 0.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing binary columns.
    binary_columns : list
        List of columns containing 'Yes'/'No' values.

    Returns:
    -------
    pd.DataFrame
        DataFrame with binary values replaced by 1 and 0.
    """
    for col in binary_columns:
        dataframe[col] = dataframe[col].map({'Yes': 1, 'No': 0})
    return dataframe

def clean_data(df,drop_nan_columns, columns_replace_nan , binary_columns):
    """
    Cleans the loaded data by handling missing values and standardizing binary values.
    
    Parameters:
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to be cleaned.
    nan_columns_to_drop : list
        List of columns to be dropped if they contain NaN values.
    nan_replacement_values : dict
        Dictionary where the keys are column names and the values are the values to replace NaN.
    binary_columns : list
        List of columns containing 'Yes' and 'No' values to be converted to 1 and 0.
    
    Returns:
    -------
    pd.DataFrame
        The cleaned DataFrame with missing values handled and binary values converted.
    """

    data_cleaned  = handle_missing_data(df,drop_nan_columns, columns_replace_nan)
    data_cleaned  = replace_binary_values(data_cleaned , binary_columns)
    return data_cleaned 



def apply_ohe(df, columns_to_onehot):
    """
    Apply One-Hot Encoding to specified categorical columns.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame with categorical columns to be encoded.
    columns_to_onehot : May be a list
        List of columns to apply One-Hot Encoding.

    Returns:
    -------
    pd.DataFrame
        DataFrame with one-hot encoded columns.
    """
    if isinstance(columns_to_onehot, str):
        columns_to_onehot = [columns_to_onehot]

    df = pd.get_dummies(df, columns=columns_to_onehot, drop_first=True)
    return df

def extract_date_features(df):
    """
    Extract date features (year, month, day) from the 'begin_date' column and drop it.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing a 'begin_date' column.

    Returns:
    -------
    pd.DataFrame
        DataFrame with new date-related columns and the original 'begin_date' column dropped.
    """
    df['begin_date'] = pd.to_datetime(df['begin_date'])
    df['año'] = df['begin_date'].dt.year
    df['mes'] = df['begin_date'].dt.month
    df['dia'] = df['begin_date'].dt.day
    df.drop('begin_date', axis=1, inplace=True)
    return df


def apply_scaler(df, numeric_columns, scaler):
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    return df

def create_feature_data(df, columns_to_onehot, scaler, columns_to_scale):
    """
    Creates new features by applying One-Hot Encoding (OHE), extracting date features, and scaling numeric columns.
    
    Parameters:
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    categorical_columns_for_ohe : list
        List of categorical columns to be one-hot encoded.
    numeric_columns_to_scale : list
        List of numeric columns to be scaled using StandardScaler.
    scaler : StandardScaler
        A pre-trained StandardScaler object to apply scaling.
    
    Returns:
    -------
    pd.DataFrame
        The DataFrame with newly created features including OHE and scaled numeric columns.
    """
    data_with_ohe  = apply_ohe(df, columns_to_onehot)
    data_with_date_features = extract_date_features(data_with_ohe)
    data_scaled  = apply_scaler(data_with_date_features, columns_to_scale, scaler)
    
    return data_scaled 





def standardize_column_names(df):
    """
    Standardize column names by converting to lowercase, replacing spaces with underscores, and removing special characters.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame with columns to be standardized.

    Returns:
    -------
    pd.DataFrame
        DataFrame with standardized column names.
    """
    df.columns = (df.columns
                  .str.strip()                        
                  .str.lower()                        
                  .str.replace(' ', '_')              
                  .str.replace(r'\(|\)', '', regex=True))  
    return df


def convert_data_types(df, column_types_dict):
    """
    Convert column data types based on the provided dictionary mapping.

    Parameters:
    ----------
    dataframe : pd.DataFrame
        The DataFrame with columns to be converted.
    column_types_dict : dict
        Dictionary where the keys are column names and the values are the target data types.

    Returns:
    -------
    pd.DataFrame
        DataFrame with columns converted to the specified data types.
    """
    for column, dtype in column_types_dict.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
    return df



def stantarize_data(df, column_types_dict):
    """
    Standardizes column names and converts data types based on the provided mapping.
    
    Parameters:
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    column_type_mapping : dict
        Dictionary where the keys are column names and the values are the target data types for conversion.
    
    Returns:
    -------
    pd.DataFrame
        The DataFrame with standardized column names and converted data types.
    """
    df_standarized = standardize_column_names(df)
    df_standarized = convert_data_types(df, column_types_dict)
    return df_standarized
