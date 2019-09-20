import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from cortex import Cortex

_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
_encoded_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month']

def encode_labels(pipeline, df):
    for c in _encoded_columns:
        encoder = LabelEncoder()
        df[c] = encoder.fit_transform(df[c])
        pipeline.set_context(f'{c}_encoder', encoder)


def bin_age(pipeline, dataframe):
    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4
    return dataframe


def bin_duration(pipeline, data):
    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5
    return data


def bin_poutcome(pipeline, df):
    df['poutcome'].replace(['nonexistent', 'failure', 'success', 'unknown', 'other'], [1,2,3,4,5], inplace  = True)


def y_dummies(pipeline, df):
    y = pd.get_dummies(df['y'], columns = ['y'], drop_first = True)
    return pd.concat([df, y], axis=1)


def train_local():
    cortex = Cortex.local('./cortex')
    builder = cortex.builder()

    train_df = pd.read_csv('./data/bank-full.csv', sep=';')
    ds = builder.dataset('default/bank-marketing-train').from_df(train_df).build()
    p = ds.pipeline('train', clear_cache=True)
    p.reset()

    p.set_context('columns', _columns)
    p.add_step(encode_labels)
    p.add_step(bin_age)
    p.add_step(bin_duration)
    p.add_step(bin_poutcome)
    p.add_step(y_dummies)

    train_df = p.run(train_df)
    print(train_df.head())
    ds.save()

    df_majority = train_df[train_df['yes'] == 0]
    df_minority = train_df[train_df['yes'] == 1]

    # Upsample to deal with imbalanced class ('yes')
    df_minority_upsampled = resample(df_minority, replace=True, n_samples= int(4640*2), random_state=123)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    y = df_upsampled['yes']

    X_train, X_test, y_train, y_test = train_test_split(df_upsampled[_columns], y, test_size = 0.1942313295, random_state = 101)
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    p.set_context('scaler', sc_X)
    X_test = sc_X.transform(X_test)

    exp = cortex.experiment('default/bank-marketing')
    exp.reset() # only need the latest run for now

    with exp.start_run() as run:
        n_estimators = 200
        run.log_param('n_estimators', n_estimators)
        run.log_param('type', 'RandomForest')

        rfc = RandomForestClassifier(n_estimators = n_estimators, n_jobs=6, random_state = 12)#criterion = entopy,gini
        rfc.fit(X_train, y_train)
    
        rfcpred = rfc.predict(X_test)
        accuracy_score = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=6, scoring = 'accuracy').mean())
        run.log_metric('accuracy', accuracy_score)

        probs = rfc.predict_proba(X_test)
        preds = probs[:,1]
        fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)
        roc_aucrfc = metrics.auc(fprrfc, tprrfc)
        run.log_metric('AUC', roc_aucrfc)

        run.log_artifact('model', rfc)
        exp.save_run(run)
        print(run)


if __name__ == "__main__":
    train_local()
