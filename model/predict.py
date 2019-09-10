import pickle
import pandas as pd
import numpy as np
import json


with open('./model/model.pk', 'rb') as f:
    model = pickle.load(f)
    clf = model['classifier']

with open('./model/scaler.pk', 'rb') as f:
    scaler = pickle.load(f)

def transform_age(df):
    df.loc[df['age'] <= 32, 'age'] = 1
    df.loc[(df['age'] > 32) & (df['age'] <= 47), 'age'] = 2
    df.loc[(df['age'] > 47) & (df['age'] <= 70), 'age'] = 3
    df.loc[(df['age'] > 70) & (df['age'] <= 98), 'age'] = 4
           
    return df


def transform_duration(df):
    df.loc[df['duration'] <= 102, 'duration'] = 1
    df.loc[(df['duration'] > 102) & (df['duration'] <= 180)  , 'duration'] = 2
    df.loc[(df['duration'] > 180) & (df['duration'] <= 319)  , 'duration'] = 3
    df.loc[(df['duration'] > 319) & (df['duration'] <= 644.5), 'duration'] = 4
    df.loc[df['duration']  > 644.5, 'duration'] = 5

    return df


def transform_poutcome(df):
    df['poutcome'].replace(['nonexistent', 'failure', 'success', 'unknown', 'other'], [1,2,3,4,5], inplace  = True)


def transform(df, model):
    transform_poutcome(df)
    transform_age(df)
    transform_duration(df)

    columns = model['columns']
    for c in columns:
        encoder = model.get(f'{c}_encoder')
        if encoder:
            df[c] = encoder.transform(df[c])

    return df


def predict(msg) -> dict:
    instances = msg.payload.get('instances', [])
    df = pd.DataFrame(columns=model['columns'], data=instances)
    x = scaler.transform(transform(df, model))
    y = clf.predict(x)

    return {'predictions': json.loads(pd.Series(y).to_json(orient='values'))}
