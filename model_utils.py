from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    # Drop unnecessary columns
    for c in ['Unnamed: 0', 'index', 'id']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Map satisfaction to binary target
    if 'satisfaction' in df.columns:
        df = df[df['satisfaction'].isin(['satisfied', 'neutral or dissatisfied'])].copy()
        df['target'] = (df['satisfaction'] == 'satisfied').astype(int)
    elif 'target' in df.columns:
        df['target'] = df['target'].astype(int)
    else:
        raise ValueError("No 'satisfaction' or 'target' column found.")

    return df.reset_index(drop=True)

def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in ['Gender','Customer Type','Type of Travel','Class'] if c in df.columns]
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['target','satisfaction']]
    return num_cols, cat_cols

def build_preprocessor(num_cols: list, cat_cols: list):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop')
