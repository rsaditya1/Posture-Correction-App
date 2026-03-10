# Posture Correction App

A real-time posture monitoring tool that uses your laptop webcam to detect whether you're sitting with good or bad posture. If you slouch for too long, it tells you to fix it.

## How it works

1. **MediaPipe** detects your body landmarks (nose, ears, shoulders) from the webcam feed
2. Joint angles and positional features are computed from those landmarks
3. An **XGBoost classifier** (exported to ONNX for speed) predicts good vs bad posture
4. The app overlays the result on your webcam feed in real time and alerts you if you've been slouching for more than 5 seconds

## Why I built this

I spend a lot of hours at my desk and wanted something lightweight that could run locally without sending video to the cloud. Most posture apps either need a phone or use some subscription service. This runs entirely on your machine.

## Tech stack

- **MediaPipe Pose Landmarker** — body landmark detection (CPU, new Tasks API)
- **XGBoost** — classification model
- **ONNX Runtime (GPU)** — fast inference on RTX 4060 via CUDAExecutionProvider
- **OpenCV** — webcam capture and display
- **FastAPI** — REST API endpoint for deployment
- **MLflow** — experiment tracking
- **scikit-learn** — preprocessing, baseline models, evaluation


## Project structure
posture-correction-app/
├── data/
│ ├── raw/ # collected CSVs (not in git)
│ ├── processed/ # train/val/test splits (not in git)
│ └── README.md # dataset documentation
├── src/
│ ├── utils.py # angle calculations, feature extraction
│ ├── collect_data.py # webcam → landmarks → CSV
│ ├── preprocess.py # cleaning, splitting
│ ├── train.py # model training with MLflow
│ ├── evaluate.py # metrics, error analysis
│ └── inference.py # real-time posture monitoring
├── tests/
│ ├── test_utils.py # unit tests for angle math
│ └── test_data.py # data quality checks
├── configs/
│ └── train_config.yaml # hyperparameters
├── api/
│ └── app.py # FastAPI endpoint
├── models/ # saved models (not in git)
├── logs/ # evaluation outputs (not in git)
├── Dockerfile
├── requirements.txt
└── README.md



## Quick start

### Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/posture-correction-app.git
cd posture-correction-app

# Create environment (Python 3.10)
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


#Collect Data
cd src
python collect_data.py --label 1 --duration 120 --fps 15  # good posture
python collect_data.py --label 0 --duration 120 --fps 15  # bad posture

