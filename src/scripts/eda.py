"""
EDA scripts

aceron for the lapidarist problem
"""

import pickle, os
import pandas as pd

from sklearn.preprocessing import LabelEncoder

paths = {
    'coords':'src/data/coords_diamonds.csv',
    'diamonds':'src/data/diamonds.csv',
    'missedDiamonds':'src/data/target.csv',
}

def read_clean_coords():
    """Return the cleaned coords dataframe"""
    coords = pd.read_csv(paths['coords']).astype(str)
    
    true_latitude = coords['latitude'].str.match('^[0-9]+$')
    latitude = coords['latitude'][true_latitude].astype(float)
    longitude = coords['longitude '][true_latitude]


    return pd.DataFrame({'longitude':longitude, 'latitude':latitude})

def load_and_encode(data, columns):
    """Load encoder objects from pickle files and apply them to the data"""
    encoders = {}
    for col in columns:
        with open(f"src/results/saves/{col}_encoder.pkl", 'rb') as f:
            encoder = pickle.load(f)
            data[col] = encoder.transform(data[col])
            encoders[col] = encoder
    return data

def encode_categorical(data, columns):
    """Encode categorical columns using LabelEncoder"""
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    return encoders

def save_encoders(encoders, directory='src/results/saves/'):
    """Save encoder objects to pickle files"""
    os.makedirs(directory, exist_ok=True)
    for col, encoder in encoders.items():
        with open(os.path.join(directory, f"{col}_encoder.pkl"), 'wb') as f:
            pickle.dump(encoder, f)

def read_clean_diamonds():
    """Return the cleaned diamonds dataframe"""
    diamonds = pd.read_csv(paths['diamonds'])
    
    diamonds.dropna(inplace=True)

    categorical_cols = ['cut', 'color', 'clarity']
    encoders = encode_categorical(diamonds, categorical_cols)
    save_encoders(encoders)

    return diamonds
