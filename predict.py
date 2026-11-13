import pandas as pd
import pickle
import uvicorn
from transformers import log_transform_subscriber_count, add_cyclical_features
from fastapi import FastAPI


app = FastAPI(title="youtube-ml-project")

with open("model.bin", "rb") as f_out:
    pipeline = pickle.load(f_out)

def predict_single(video: dict):
    df = pd.DataFrame([video])
    result = pipeline.predict_proba(df)[0, 1]
    return float(result)

@app.post("/predict")
def predict(video: dict):
    prediction = predict_single(video)

    return {"viral_probability": prediction, 
            "is_viral": prediction > 0.5}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
