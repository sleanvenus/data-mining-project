import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LABEL = 'exceeds50K'
DISCRETE_COLS = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'sex', 'native-country']
CONTINOUS_COLS = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

def normalize_data(x1):
    scaler = StandardScaler()
    for col in x1.columns.tolist():
        norm_vals = scaler.fit_transform(x1[col].values.reshape((-1, 1)))
        x1[col] = norm_vals
    return x1

def margin_prob(df):
    df_norm = df.copy()
    prob = {}

    for col in DISCRETE_COLS:  # ['native-country']:
        prob[col] = df_norm.groupby(col).apply(lambda x: np.sum(x[LABEL] == 1) / x[LABEL].shape[0])
        df_norm[col] = df_norm[col].apply(lambda x: prob[col].T[x])

    y1 = df_norm[LABEL]
    x1 = df_norm.drop(columns=LABEL)
    # x1 = x1.drop(columns='relationship')
    # x1.drop(columns='fnlwgt', inplace=True)
    # x1.drop(columns='native-country', inplace=True)
    x1 = normalize_data(x1)
    return x1, y1, prob

def margin_rank(df):
    df_norm = df.copy()
    discrete_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'sex', 'native-country']
    rank = {}

    for col in DISCRETE_COLS:  # ['native-country']:
        rank[col] = df_norm.groupby(col).apply(lambda x: np.sum(x[LABEL] == 1) / x[LABEL].shape[0])
        rank[col] = rank[col].rank()
        df_norm[col] = df_norm[col].apply(lambda x: rank[col].T[x])

    y1 = df_norm[LABEL]
    x1 = df_norm.drop(columns=LABEL)
    x1 = normalize_data(x1)
    return x1, y1, rank

def one_hot(df):
    df_norm = df.copy()
    y1 = df_norm[LABEL]
    x1 = df_norm.drop(columns=LABEL)
    x1 = pd.get_dummies(x1)

    return x1, y1

def one_hot_cut(df):
    df_norm = df.copy()
    prob = {}

    for col in ['native-country']:  # ['native-country']:
        prob[col] = df_norm.groupby(col).apply(lambda x: np.sum(x[LABEL] == 1) / x[LABEL].shape[0])
        df_norm[col] = df_norm[col].apply(lambda x: prob[col].T[x])

    for col in CONTINOUS_COLS + ['native-country']:
        df_norm[col] = pd.cut(df_norm[col], 10, labels=False, duplicates='drop')
        df_norm[col] = df_norm[col].astype(str)
    y1 = df_norm[LABEL]
    x1 = df_norm.drop(columns=LABEL)
    x1 = pd.get_dummies(x1).astype(np.int)

    return x1, y1, prob

def one_hot_qcut(df):
    df_norm = df.copy()
    prob = {}

    for col in ['native-country']:  # ['native-country']:
        prob[col] = df_norm.groupby(col).apply(lambda x: np.sum(x[LABEL] == 1) / x[LABEL].shape[0])
        df_norm[col] = df_norm[col].apply(lambda x: prob[col].T[x])

    for col in CONTINOUS_COLS + ['native-country']:
        df_norm[col] = pd.qcut(df_norm[col], 10, labels=False, duplicates='drop')
        df_norm[col] = df_norm[col].astype(str)
    y1 = df_norm[LABEL]
    x1 = df_norm.drop(columns=LABEL)
    x1 = pd.get_dummies(x1).astype(np.int)

    return x1, y1, prob