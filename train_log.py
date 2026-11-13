import numpy as np
import pandas as pd
import kagglehub
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import log_transform_subscriber_count, add_cyclical_features


# Parameters for model training

C=0.01
random_state = 42
 
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

    df.drop(['video_id', 'views', 'shares', 'comments', 'likes'], axis=1, inplace=True)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["viral"])
    
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    y_train = df_train["viral"]
    y_test = df_test["viral"]

    df_train = df_train.drop(columns=["viral"])
    df_test = df_test.drop(columns=["viral"])

    return df_train, df_test, y_train, y_test


# Define numerical features (left out hour_sin and hour_cos, as they are already scaled)
numerical_features = ['duration_sec', 'title_length', 'tags_count', 'description_length', 'log_subscriber_count']

# Define categorical features (left out thumbnail_face, as it is already naturally a 0/1 feature)
categorical_features = ['upload_day', 'category']

def train(df_train, y_train, df_test, y_test, C=0.01, random_state=42):

    # 1. Build the ColumnTransformer
    preprocessor = make_column_transformer(
        (StandardScaler(), numerical_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
        n_jobs=-1, remainder="passthrough"
    )

    # 2. Build the Pipeline
    pipeline = make_pipeline(
        FunctionTransformer(log_transform_subscriber_count),
        FunctionTransformer(add_cyclical_features),
        preprocessor,
        LogisticRegression(C=C, max_iter=1000, random_state=random_state),
    )

    pipeline.fit(df_train, y_train)
    y_pred = pipeline.predict_proba(df_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return pipeline, auc


def save_model(pipeline, filename="model.bin"):
    with open(filename, "wb") as f_out:
        pickle.dump(pipeline, f_out)


if __name__ == "__main__":
    
    df = load_dataset()
    df_train, df_test, y_train, y_test = drop_split_dataset(df)

    # Fit the entire pipeline on training data and test
    pipeline, auc = train(df_train, y_train, df_test, y_test)

    # Save the trained model
    save_model(pipeline)

    print(f"Test ROC AUC: {auc:.4f}")
    print('Model saved to model.bin')