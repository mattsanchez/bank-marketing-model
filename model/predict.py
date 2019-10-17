import logging
import json
import pandas as pd
import random
from cortex import Cortex, Message

random.seed(7113)

log = logging.getLogger()

cortex = Cortex.local('./cortex')
builder = cortex.builder()
exp = cortex.experiment('default/bank-marketing')

def load_model(exp, filter, sort=None):
    if not sort:
        sort = {"endTime": -1}

    # HACK for local mode experiments
    runs = exp.runs()
    for run in runs:
        if run.get_param('type') == filter['params.type']:
            return run.get_artifact('model')
    
    raise Exception(f'Model not found for selection filter: {filter}')
    
    # the below only works with remote experiments
    # runs = exp.find_runs(filter, sort, 1)
    # if len(runs) == 0:
        # raise Exception(f'Model not found for selection filter: {filter}')

    # run = runs[0]
    # return run.get_artifact('model')


rfc = load_model(exp, {"params.type": "RandomForest"})
dt = load_model(exp, {"params.type": "DecisionTree"})

train_ds = cortex.dataset('default/bank-marketing-train')

# Setup predict pipeline by copying the train pipeline and dropping the y_dummies step
train_pipeline = train_ds.pipeline('train')
pipeline = builder.pipeline('predict')
pipeline.from_pipeline(train_pipeline)
pipeline.remove_step('encode_labels')   # Replace with step that uses encoders from training
pipeline.remove_step('y_dummies')       # Remove all steps that deal with target variable

def encode_columns(pipeline, df):
    columns = pipeline.get_context('columns')
    for c in columns:
        encoder = pipeline.get_context(f'{c}_encoder')
        if encoder:
            df[c] = encoder.transform(df[c])
        

pipeline.add_step(encode_columns)

def do_predict(msg: Message, clf) -> dict:
    instances = msg.payload.get('instances', [])
    log.info(f'Instances: {instances[0:5]}')

    df = pd.DataFrame(columns=pipeline.get_context('columns'), data=instances)
    
    # Prepare model frame for predition using the training pipeline
    df = pipeline.run(df)

    # Use the same scaler transform used on the training data
    scaler = pipeline.get_context('scaler')
    
    x = scaler.transform(df)
    y = clf.predict(x)
    
    return {'predictions': json.loads(pd.Series(y).to_json(orient='values'))}


def predict_rfc(msg: Message, model_context: dict) -> dict:
    return do_predict(msg, rfc)

def predict_dt(msg: Message, model_context: dict) -> dict:
    return do_predict(msg, dt)


if __name__ == "__main__":
    with open('./test/2-instances.json') as f:
        test_json = json.load(f)
    msg = Message(test_json)
    result = predict_rfc(msg)
    print(result)
