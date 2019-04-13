# little helpers I keep reusing
import pandas as pd
import numpy as np

def quick_summary(df):
    print('shape:', df.shape)
    print('dtypes:')
    print(df.dtypes)
    print('nulls per col:')
    print(df.isnull().sum())
    print('memory mb:', df.memory_usage().sum() / 1024 / 1024)

def split_xy(df, target):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
