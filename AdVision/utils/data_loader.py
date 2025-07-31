import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(uploaded_file):
    """
    Load CSV file into a DataFrame.
    """
    df = pd.read_csv(uploaded_file)
    return df


def preprocess_data(df):
    """
    Clean and preprocess the dataset:
    - Drop duplicates
    - Handle missing values
    - Label encode 'influencer' column
    - Ensure correct column types
    """

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()  # or use df.fillna() with strategy if preferred

    # Ensure lowercase column names
    df.columns = [col.lower() for col in df.columns]

    # Label Encoding for 'influencer' column
    if 'influencer' in df.columns:
        le = LabelEncoder()
        df['influencer'] = le.fit_transform(df['influencer'])

    # Ensure numeric types for predictors
    numeric_cols = ['tv', 'radio', 'social_media', 'sales']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Final cleaning
    df = df.dropna()

    return df