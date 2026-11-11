import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import average_precision_score
import pickle


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

    df.drop(['views', 'shares', 'comments', 'likes'], axis=1, inplace=True)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["viral"])
    
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    y_train = df_train["viral"]
    y_test = df_test["viral"]

    df_train = df_train.drop(columns=["viral"])
    df_test = df_test.drop(columns=["viral"])

    return df_train, df_test, y_train, y_test


# 1. Feature engineering functions
def log_transform_subscriber_count(df):
    df = df.copy()
    df["log_subscriber_count"] = np.log1p(df["subscriber_count"])
    return df.drop(columns=["subscriber_count"])


def add_cyclical_features(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["upload_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["upload_hour"] / 24)
    return df.drop(columns=["upload_hour"])


def df_to_dict(df):
    return df.to_dict(orient="records")


# 2. Define numerical columns to scale
# Note: 'subscriber_count' becomes 'log_subscriber_count' before scaling runs
numerical_features = ['duration_sec', 'title_length', 'tags_count', 'description_length', 'log_subscriber_count']

scaler_step = ColumnTransformer(
    transformers=[("num", StandardScaler(), numerical_features)],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")


# 3. Build the pipeline
pipeline = Pipeline([
    ("log_transform", FunctionTransformer(log_transform_subscriber_count)),
    ("cyclical_features", FunctionTransformer(add_cyclical_features)),
    ("scaler", scaler_step),
    ("to_dict", FunctionTransformer(df_to_dict)),
    ("vectorizer", DictVectorizer(sparse=False)),
    ("model", LogisticRegression(C=C, max_iter=1000, random_state=random_state)),
])

def save_model(pipeline, filename="model.bin"):
    with open(filename, "wb") as f_out:
        pickle.dump(pipeline, f_out)

# 4. Train and evaluate (assuming df_train, y_train, df_test, y_test exist)
if __name__ == "__main__":
    df = load_dataset()
    df_train, df_test, y_train, y_test = drop_split_dataset(df)

    # Fit the entire pipeline on training data
    pipeline.fit(df_train, y_train)

    # Predict / evaluate on test data
    y_pred = pipeline.predict_proba(df_test)[:, 1]
    aucpr = average_precision_score(y_test, y_pred)

    # Save the trained model
    save_model(pipeline)

    print(f"Test AUC-PR: {aucpr:.4f}")
    print('Model saved to model.bin')
    
    
