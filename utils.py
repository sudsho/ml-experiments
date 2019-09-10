# little helpers I keep reusing
import pandas as pd
import numpy as np

def quick_summary(df):
    print('shape:', df.shape)
    print('dtypes:')
    print(df.dtypes)
    print('nulls per col:')
    print(df.isnull().sum())
    print('memory mb:', df.memory_usage(deep=True).sum() / 1024 / 1024)

def split_xy(df, target):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

def value_counts_pct(s):
    out = pd.concat([s.value_counts(), s.value_counts(normalize=True).round(3)], axis=1)
    out.columns = ['n', 'pct']
    return out

def reduce_mem(df):
    for c in df.select_dtypes(include=['int64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='integer')
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = pd.to_numeric(df[c], downcast='float')
    return df

def high_cardinality(df, threshold=50):
    out = {}
    for c in df.select_dtypes(include=['object']).columns:
        n = df[c].nunique()
        if n > threshold:
            out[c] = n
    return out
