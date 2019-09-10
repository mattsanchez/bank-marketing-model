import pandas as pd
from model.predict import predict
import warnings
warnings.filterwarnings('ignore')


def test_predict_1():
    # Run a single instance through predict
    df = pd.read_csv('./test/1-instance.csv', sep = ';')
    
    # Use only the columns the trained model expects (drops the outcome column)
    x = df.iloc[:, :-1]
    
    # Convert the outcome column to 0 and 1
    y = df['y'].copy()
    y.loc[y == 'no'] = 0
    y.loc[y == 'yes'] = 1

    instances = list(x.values)
    predictions = predict({'payload': {'instances': instances}})
    
    # Check that our model made a correct prediction
    assert y[0] == predictions['predictions'][0]

    print(f'predictions: {predictions}')
