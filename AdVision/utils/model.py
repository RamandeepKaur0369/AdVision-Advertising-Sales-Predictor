import joblib
import pandas as pd

def load_model(path='models/mlr_model.joblib'):
    """
    Load a trained model from a .joblib file.
    
    Parameters:
        path (str): Path to the saved model file.
        
    Returns:
        model: The loaded machine learning model.
    """
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        raise Exception(f"Model file not found at: {path}")


def predict(model, new_data):
    """
    Predict target values using the loaded model.
    
    Parameters:
        model: A trained machine learning model.
        new_data (DataFrame): Input features as a pandas DataFrame.
        
    Returns:
        np.ndarray: Predicted values.
    """
    return model.predict(new_data)
