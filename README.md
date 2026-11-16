# YouTube Video Virality Prediction

An end-to-end machine learning project that predicts whether a YouTube video will go **viral** *before* it is published, using pre-upload video metadata and channel statistics.

---

## Table of Contents
- [Problem Description](#problem-description)
- [Dataset & Data Leakage Prevention](#dataset--data-leakage-prevention)
- [Exploratory Data Analysis (EDA) Summary](#exploratory-data-analysis-eda-summary)
- [Model Training & Selection](#model-training--selection)
- [Project Structure](#project-structure)
- [Dependency & Environment Management](#dependency--environment-management)
- [Reproducibility: Data & Model Training](#reproducibility-data--model-training)
- [Local Deployment (FastAPI)](#local-deployment-fastapi)
- [Containerization (Docker)](#containerization-docker)
- [Cloud Deployment (Fly.io)](#cloud-deployment-flyio)

---

## Problem Description

Content creators, media agencies, and digital marketers invest significant time and budget producing YouTube videos without knowing how well a video's packaging and timing will perform. While post-publish analytics tell creators how a video *already* performed, they cannot guide decisions **prior to publishing**.

### Objective & Practical Use Case
This project builds and deploys a binary classification service that estimates the **probability of a YouTube video going viral** (defined as reaching the top $20\%$ of viewership) strictly from **pre-upload attributes**:
- **Content & Packaging:** Video category, video duration (`duration_sec`), title character length (`title_length`), presence of a human face in the thumbnail (`thumbnail_face`), number of tags (`tags_count`), and description length (`description_length`).
- **Publishing Schedule:** Hour of upload (`upload_hour`) and day of the week (`upload_day`).
- **Channel Reach:** Current channel subscriber base (`subscriber_count`).

**How the solution is used:** Creators or scheduling tools can query the deployed REST API (`/predict`) prior to publishing to evaluate video metadata configurations, test different upload times or packaging strategies, and prioritize promotional spend on high-potential uploads.

---

## Dataset & Data Leakage Prevention

- **Source:** [Kaggle — What Makes a YT Video Go Viral? (12,000 Vids)](https://www.kaggle.com/datasets/vedauv/what-makes-a-yt-video-go-viral-12000-vids) (`vedauv/what-makes-a-yt-video-go-viral-12000-vids`)
- **Size:** $12,000$ records across $15$ columns (`0` missing values, `0` duplicate rows).
- **Automatic Download:** Both `notebook.ipynb` and `train.py` automatically fetch the dataset into `./dataset/viral_videos.csv` using `kagglehub`, ensuring effortless reproducibility without manual downloads.

### Preventing Target Leakage
To ensure the model is valid for real-world pre-upload inference, the following columns are removed prior to training:
1. `video_id`: Unique identifier carrying no predictive signal.
2. `views`: Directly defines the target label `viral` (top $20\%$ most-viewed videos).
3. `likes`, `comments`, `shares`: Post-upload engagement metrics that are unavailable before a video goes live and exhibit severe multicollinearity ($r > 0.93$ among one another).

---

## Exploratory Data Analysis (EDA) Summary

Full Exploratory Data Analysis is documented in `notebook.ipynb`, covering:
- **Target Variable Analysis:** Mild-to-moderate class imbalance with **$80.0\%$ non-viral (`0`)** and **$20.0\%$ viral (`1`)** videos. Since the dataset is only mildly to moderately imbalanced (20% positive class), ROC-AUC provides a reliable measure of the model's ability to distinguish between positive and negative classes. PR-AUC is more sensitive to false positives on the minority class, usually used in 1-10% positive class scenarios. PR-AUC gains more importance in highly imbalanced datasets, where ROC AUC is very distorted.
- **Numerical Distributions & Skewness:** Social media reach metrics (`subscriber_count`, `views`, `likes`, `comments`, `shares`) exhibit heavy right-skewness. Applying a $\log(1 + x)$ transformation (`np.log1p`) to `subscriber_count` normalizes its distribution and substantially strengthens its linear correlation with `viral`. Video `duration_sec` shows a negative correlation ($r = -0.165$) with virality, indicating shorter-to-medium videos go viral more frequently than very long uploads.
- **Categorical Risk Ratio & Mutual Information:**
  - `category` carries by far the highest mutual information score ($0.0453$). Risk ratio analysis reveals that **Music** ($\text{risk} = 2.91\times$) and **Comedy** ($\text{risk} = 1.83\times$) have significantly elevated virality rates compared to the global $20\%$ baseline, whereas **Vlog** ($0.35\times$) and **Travel** ($0.38\times$) have the lowest.
  - `thumbnail_face` provides a modest positive lift ($\text{risk} = 1.13\times$ when a face is present vs. $0.91\times$ without).
  - `upload_hour` and `upload_day` show subtle cyclical variations, peaking late at night (`upload_hour = 23`, $\text{risk} = 1.20\times$).

---

## Model Training & Selection

In `notebook.ipynb`, the dataset was split into $80\%$ training ($9,600$ samples) and $20\%$ hold-out testing ($2,400$ samples) using stratified sampling. Feature engineering was tailored to each model family:
- **Linear Models:** $\log(1 + x)$ transformation on `subscriber_count`, cyclical trigonometric encoding ($\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)$ and $\cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$) on `upload_hour`, `StandardScaler` on numerical features, and one-hot encoding on categorical features.
- **Tree-Based Models:** Interaction binning of `upload_day` and `upload_hour` into `day_time_bin` alongside one-hot encoding.

Four model families were trained and tuned across **5-fold Stratified Cross-Validation**:

| Model | Tuned Hyperparameters | 5-Fold CV Mean ROC-AUC |
| :--- | :--- | :---: |
| **Logistic Regression** *(Selected)* | `C=0.01`, `max_iter=1000` | **0.9065** |
| **Decision Tree** | `max_depth=4`, `min_samples_leaf=100` | 0.9008 |
| **Random Forest** | `n_estimators=170`, `max_depth=10`, `min_samples_leaf=10` | 0.8972 |
| **XGBoost** | `eta=0.05`, `max_depth=3`, `min_child_weight=30`, `num_boost_round=83` | 0.9068 |

### Why Logistic Regression Was Chosen
Regularized Logistic Regression (`C=0.01`) outperformed Decision Trees and Random Forests and virtually tied tuned XGBoost. When a linear model and a gradient-boosted tree ensemble converge to the same validation ceiling, it demonstrates that the underlying feature-to-target relationship is predominantly linear/additive. Logistic Regression was selected for production deployment due to its simplicity, fast inference latency, tiny memory footprint, and high interpretability—achieving a **Test ROC-AUC of `0.9039`** on the unseen hold-out set.

### Note on model comparison
A PR-AUC comparison (not featured) has shown very close results, further supporting the conclusion that both models are capturing the relevant signal for the minority class. In practice, the choice of Logistic Regression is justified by its operational advantages in production settings. 

---

## Project Structure

```text
.
├── dataset/
│   └── viral_videos.csv       # Automatically downloaded via kagglehub
├── notebook.ipynb             # Data validation, extensive EDA, CV tuning & model comparison
├── transformers.py            # Custom feature engineering functions used in the sklearn Pipeline
├── train.py                   # Standalone script to download data, train pipeline & export model.bin
├── predict.py                 # FastAPI inference web service exposing POST /predict
├── predict_test.py            # Client script to test local or cloud-deployed API endpoints
├── model.bin                  # Serialized scikit-learn preprocessing + LogisticRegression pipeline
├── pyproject.toml             # Project dependencies and configuration managed via uv
├── uv.lock                    # Locked deterministic dependency versions
├── Dockerfile                 # Container definition for production deployment
└── README.md                  # Project documentation
```

---

## Dependency & Environment Management

This project uses **[uv](https://docs.astral.sh/uv/)** with `pyproject.toml` (Python $\ge 3.11$) for fast, reproducible virtual environment and dependency management.

### 1. Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository & Install Dependencies
Running `uv sync` creates a `.venv` virtual environment in the project root and installs all locked production and development dependencies:
```bash
git clone <YOUR_REPO_URL>
cd youtube-ml-project
uv sync
```

### 3. Activate the Virtual Environment
```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
*(Alternatively, prefix any command with `uv run`, e.g., `uv run python train.py`, without manually activating the shell).*

---

## Reproducibility: Data & Model Training

No manual data download is required—`kagglehub` automatically downloads and extracts `viral_videos.csv` into `./dataset/`.

### Re-running the Training Script
To train the final scikit-learn pipeline from scratch, evaluate it on the hold-out test split, and generate `model.bin`:
```bash
python train.py
```
Expected output:
```text
Test ROC AUC: 0.9039
Model saved to model.bin
```

### Re-running the Notebook
With the virtual environment activated, launch Jupyter to inspect or re-execute all cells in `notebook.ipynb`:
```bash
jupyter notebook notebook.ipynb
```

---

## Local Deployment (FastAPI)

The trained pipeline (`model.bin`) is served via **FastAPI** and **Uvicorn** in `predict.py`.

### 1. Start the Local Web Service
```bash
python predict.py
```
*(Or directly with Uvicorn: `uvicorn predict:app --host 0.0.0.0 --port 8000`)*

The service will listen at `http://0.0.0.0:8000` (interactive Swagger UI docs are also available at `http://localhost:8000/docs`).

### 2. Test the Local Endpoint
Open a second terminal (with the virtual environment activated) and run `predict_test.py`:
```bash
python predict_test.py
```
Or send a `curl` request directly:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "category": "Education",
           "duration_sec": 1201,
           "title_length": 72,
           "thumbnail_face": 0,
           "upload_hour": 13,
           "upload_day": "Sat",
           "tags_count": 15,
           "description_length": 569,
           "subscriber_count": 2789543
         }'
```

Expected JSON response:
```json
{
  "viral_probability": 0.5832448669520311,
  "is_viral": true
}
```

---

## Containerization (Docker)

The application is fully containerized so it can run identically across any machine or cloud provider.

### 1. Build the Docker Image
From the project root directory, run:
```bash
docker build -t youtube-ml-project .
```

### 2. Run the Docker Container
Map container port `8000` to host port `8000`:
```bash
docker run -it --rm -p 8000:8000 youtube-ml-project
```

### 3. Test the Containerized API
Run the test script in another terminal while the container is active:
```bash
python predict_test.py
```

---

## Cloud Deployment (Fly.io)

The containerized FastAPI service is deployed to the cloud using **[Fly.io](https://fly.io/)**.

- **Live Endpoint URL:** `https://youtube-ml-project.fly.dev/predict`
- **Interactive API Docs:** `https://youtube-ml-project.fly.dev/docs`

### Step-by-Step Deployment Commands
To deploy this service to your own Fly.io account:

1. **Install `flyctl` and authenticate:**
   ```bash
   # macOS / Linux
   curl -L https://fly.io/install.sh | sh
   fly auth login
   ```

2. **Initialize the Fly.io app** (generates `fly.toml` using the project `Dockerfile` on internal port `8000`):
   ```bash
   fly launch --no-deploy
   ```

3. **Build and deploy to the cloud:**
   ```bash
   fly deploy
   ```

4. **Check deployment status and logs:**
   ```bash
   fly status
   fly logs
   ```

### Testing the Live Cloud Deployment

You can test the live cloud endpoint in two ways:

**Option A — Using `curl`:**
```bash
curl -X POST "https://youtube-ml-project.fly.dev/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "category": "Education",
           "duration_sec": 1201,
           "title_length": 72,
           "thumbnail_face": 0,
           "upload_hour": 13,
           "upload_day": "Sat",
           "tags_count": 15,
           "description_length": 569,
           "subscriber_count": 2789543
         }'
```

**Option B — Using `predict_test.py`:**
Uncomment the deployment URL in `predict_test.py`:
```python
url = "https://youtube-ml-project.fly.dev/predict"
```
Then run:
```bash
python predict_test.py
```