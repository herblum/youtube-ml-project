
import pandas as pd
import numpy as np
import kagglehub
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline


# Parameters for model training

eta = 0.01
max_depth = 3
min_child_weight = 1
n_estimators = 300
eval_metric = 'auc'

 
# Load the dataset

def load_dataset():

    path = kagglehub.dataset_download(
        "vedauv/what-makes-a-yt-video-go-viral-12000-vids",
        output_dir="./dataset",
        force_download=True
     )

    df = pd.read_csv("./dataset/viral_videos.csv")

    return df


# Data preparation

def drop_split_dataset(df):

    df.drop(['views', 'shares', 'comments', 'likes'], axis=1, inplace=True)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["viral"])
    
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    y_train = df_train["viral"]
    y_test = df_test["viral"]

    df_train = df_train.drop(columns=["viral"])
    df_test = df_test.drop(columns=["viral"])

    return df_train, df_test, y_train, y_test


# Model training and pipeline creation

def train_xgb(df_train, y_train):

    train_dicts = df_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)    

    xgb_model = xgb.XGBClassifier()

    xgb_model.set_params(eta=eta,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        n_estimators=n_estimators,
                        eval_metric=eval_metric,
                        nthread=8,
                        seed=42,
                        verbosity=1)

    pipeline = make_pipeline(DictVectorizer(), xgb_model)

    pipeline.fit(train_dicts, y_train)

    return pipeline


# Test the model

def test_model(pipeline, df_test, y_test):

    test_dicts = df_test.to_dict(orient='records')
    y_pred_proba = pipeline.predict_proba(test_dicts)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return roc_auc

# Saving the Model

def save_model(pipeline, filename="model.bin"):
    with open(filename, "wb") as f_out:
        pickle.dump(pipeline, f_out)


if __name__ == "__main__":

    df = load_dataset()
    df_train, df_test, y_train, y_test = drop_split_dataset(df)
    pipeline = train_xgb(df_train, y_train)
    roc_auc = test_model(pipeline, df_test, y_test)
    save_model(pipeline)

    print(f'Model ROC AUC: {roc_auc:.4f}')
    print('Model saved to model.bin')