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
run = exp.last_run()
if not run:
    raise Exception('Model has not been trained locally yet - run not found.')

train_ds = cortex.dataset('default/bank-marketing-train')

# Setup predict pipeline by copying the train pipeline and dropping the y_dummies step
train_pipeline = train_ds.pipeline('train')
pipeline = builder.pipeline('predict')
pipeline.from_pipeline(train_pipeline)
pipeline.remove_step('encode_labels')
pipeline.remove_step('y_dummies')

def encode_columns(pipeline, df):
    columns = pipeline.get_context('columns')
    for c in columns:
        encoder = pipeline.get_context(f'{c}_encoder')
        if encoder:
            df[c] = encoder.transform(df[c])
        

pipeline.add_step(encode_columns)


def predict(msg: Message) -> dict:
    instances = msg.payload.get('instances', [])
    log.info(f'Instances: {instances[0:5]}')

    df = pd.DataFrame(columns=pipeline.get_context('columns'), data=instances)
    
    # Prepare model frame for predition using the training pipeline
    df = pipeline.run(df)

    # Get the classifier from the run
    clf = run.get_artifact('model')

    # Use the same scaler transform used on the training data
    scaler = pipeline.get_context('scaler')
    
    x = scaler.transform(df)
    y = clf.predict(x)
    
    return {'predictions': json.loads(pd.Series(y).to_json(orient='values'))}


if __name__ == "__main__":
    with open('./test/1-instance.json') as f:
        test_json = json.load(f)
    msg = Message(test_json)
    result = predict(msg)
    print(result)
