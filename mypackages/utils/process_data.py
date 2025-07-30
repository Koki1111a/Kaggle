import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ProcessData:
    def __init__(self):
        self.encoders = {}

    def drop_features(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        return df.drop(features, axis=1)

    def get_features(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        return df[features]

    def encode_features(self, df: pd.DataFrame, features: list = None) -> pd.DataFrame:
        if not features:
            features = df.select_dtypes(include=['object']).columns.tolist()
        for feature in features:
            if df[feature].dtype == 'object':
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature])
                self.encoders[feature] = le
        return df

    def decode_features(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        if not features:
            features = df.select_dtypes(include=['object']).columns.tolist()
        for feature in features:
            if feature in self.encoders:
                le = self.encoders[feature]
                df[feature] = le.inverse_transform(df[feature])
        return df