import pandas as pd
import numpy as np
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def clear_missing(df: pd.DataFrame):
    result_df = df
    row_count = df.shape[0]
    missing_row_count = df.isna().any(axis=1).sum()
    if(missing_row_count <= math.floor(0.1 * row_count)):
        result_df.dropna(inplace=True)
    else:
        imputer = SimpleImputer(fill_value=np.nan, strategy='mean')
        X = imputer.fit_transform(result_df)
        result_df = pd.DataFrame(X, columns=result_df.columns)
    return result_df


def encode_cols(df: pd.DataFrame, col_names):
    l1 = LabelEncoder()
    for col in col_names:
        df[col] = l1.fit_transform(df[col])
    return df


def drop_cols(df: pd.DataFrame, col_names):
    df.drop_duplicates(inplace=True)
    for col in col_names:
        df.drop(col, axis=1, inplace=True)
    return df


def scale_dataset(df: pd.DataFrame, no_scale):
    scaler = StandardScaler()
    for col in df.columns:
        if col not in no_scale:
            df[[col]] = scaler.fit_transform(df[[col]])
    return df
