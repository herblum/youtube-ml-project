import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel 


app = FastAPI(title="youtube-ml-project")

with open("model.bin", "rb") as f_out:
    pipeline = pickle.load(f_out)

def predict_single(video: dict):
    result = pipeline.predict_proba([video])[0, 1]
    return float(result)

@app.post("/predict")
def predict(video: dict):
    prediction = predict_single(video)

    return {"viral_probability": prediction, 
            "is_viral": prediction > 0.5}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
