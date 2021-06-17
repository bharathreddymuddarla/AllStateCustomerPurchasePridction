import pandas as pd
from sklearn.preprocessing import LabelEncoder

def datawrangle(df):
    print('**** entered datawrangling ****')
    train_df = df
    risk_factor_mean = round(train_df['risk_factor'].mean())
    train_df['risk_factor'] = train_df['risk_factor'].fillna(round((train_df['risk_factor'].mean())))
    # Droping rest of records having nan values
    train_df = train_df.dropna()
    # Renaming column names - converting uppercase names to lower case
    train_df.rename(
        columns={'customer_ID': 'customer_id', 'C_previous': 'c_previous', 'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd',
                 'E': 'e', 'F': 'f', 'G': 'g'}, inplace=True)
    train_df = train_df.drop('time', axis=1)
    # dropping customer_id column
    train_df = train_df.drop('customer_id', axis=1)
    # dropping day column
    train_df = train_df.drop('day', axis=1)
    train_df['duration_previous'] = train_df['duration_previous'].astype('int')
    categorical_cols = ['state', 'location', 'group_size', 'homeowner', 'car_value', 'risk_factor', 'married_couple']
    for col in categorical_cols:
        train_df[col] = train_df[col].astype('category')
    # instantiate labelencoder object
    le = LabelEncoder()
    # apply le on categorical feature columns
    train_df[categorical_cols] = train_df[categorical_cols].apply(lambda col: le.fit_transform(col))


    return train_df