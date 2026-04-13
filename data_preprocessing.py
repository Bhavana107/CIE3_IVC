import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Handle missing values
    df = df.dropna()  # or fillna if used
    return df

def encode_data(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def preprocess(path):
    df = load_data(path)
    df = clean_data(df)
    df = encode_data(df)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = scale_data(X)

    return X, y