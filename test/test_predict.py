import pandas as pd
from model.predict import predict
from cortex import Cortex, Message
import warnings
warnings.filterwarnings('ignore')


def test_predict_1():
    # Run a single instance through predict
    df = pd.read_csv('./test/1-instance.csv', sep=';')

    # Use only the columns the trained model expects (drops the outcome column)
    x = df.iloc[:, :-1]

    # Convert the outcome column to 0 and 1
    y = df['y'].copy()
    y.loc[y == 'no'] = 0
    y.loc[y == 'yes'] = 1

    instances = list(x.values)
    predictions = predict(Message({'payload': {'instances': instances}}))
    print(f'Predictions: {predictions}')

    # Check that our model made a correct prediction
    assert y[0] == predictions['predictions'][0]


def test_predict_2():
    # Run a single instance through predict
    df = pd.read_csv('./test/2-instances.csv', sep=';')

    # Use only the columns the trained model expects (drops the outcome column)
    x = df.iloc[:, :-1]

    # Convert the outcome column to 0 and 1
    y = df['y'].copy()
    y.loc[y == 'no'] = 0
    y.loc[y == 'yes'] = 1

    instances = list(x.values)
    predictions = predict(Message({'payload': {'instances': instances}}))
    print(f'Predictions: {predictions}')

    # Check that our model made the correct predictions
    assert y[0] == predictions['predictions'][0]
    assert y[1] == predictions['predictions'][1]


def test_predict_3():
    instances = [
        [58,"management","married","tertiary", "no",2143,"yes","no","unknown",5,"may",261,1,-1,0,"unknown"],
        [51.0,"technician","married","primary", "no",-459,"yes", "yes", "cellular",5,"may",261,1,-1,0,"unknown"],
        [41,"technician","married","secondary","no",1270,"yes","no","unknown",5,"may",1389,1,-1,0,"unknown"]
    ]

    y = [0, 0, 1]

    response = predict(Message({'payload': {'instances': instances}}))
    predictions = response['predictions']
    print(f'Predictions: {predictions}')

    # Check that our model made the correct predictions
    assert y == predictions
