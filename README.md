---
title: Parkinson's Motor Impairment Predictor
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Parkinson's Motor Impairment Score API

This project is a **FastAPI** service that predicts Parkinson's-related motor impairment from patient drawing images.

It accepts two drawing styles:

- **Wave drawings**
- **Spiral drawings**

Each image is preprocessed and passed into a deep learning model to produce:

- a raw model logit
- a sigmoid probability
- a normalized motor impairment score from 0 to 100
- a severity label
- a human-readable description

The project is designed for **local development**, **Docker deployment**, and **Hugging Face Spaces**.


The service is intended to:

1. receive an uploaded image
2. preprocess it consistently
3. run inference using a pretrained model
4. return a structured prediction response

---

## Key Features

### 1. FastAPI-based service

The API uses FastAPI for fast request handling, automatic validation, and interactive documentation.

### 2. Two prediction routes

- Wave drawings use a **VGG19-based** model
- Spiral drawings use a **ResNet101-based** model

### 3. Startup model loading

Both models are loaded during application startup so the first request is faster.

### 4. Custom preprocessing pipeline

The image pipeline reproduces the original training preprocessing using:

- OpenCV
- NumPy
- Pillow

### 5. Hugging Face model download

The trained `.h5` models are downloaded from Hugging Face Hub when needed.

### 6. CORS support

The API is configured to accept cross-origin requests from browser-based clients.

### 7. Docker-ready deployment

The repository includes a Dockerfile for Hugging Face Spaces deployment.

---

## Repository Structure

- [main.py](main.py) - FastAPI application entry point and API routes
- [services/predictor.py](services/predictor.py) - model loading, preprocessing, and prediction logic
- [services/**init**.py](services/__init__.py) - package initializer
- [requirements.txt](requirements.txt) - Python dependencies
- [Dockerfile](Dockerfile) - container configuration for deployment
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) - GitHub Actions workflow for deployment to Hugging Face
- [test/](test/) - sample images for testing and experimentation

---

## How the API Works

### Request flow

1. A client uploads a drawing image using multipart form data
2. The API validates that the file is an image
3. The image is converted into a consistent tensor-like NumPy array
4. The correct model is loaded if not already cached
5. The model returns a raw logit
6. The logit is converted to a sigmoid probability
7. The result is normalized into a motor impairment score
8. A severity label and description are returned

### Preprocessing pipeline

The preprocessing logic performs the following steps:

1. load the image
2. convert to grayscale
3. apply Otsu thresholding with inversion
4. resize to 224 × 224
5. replicate grayscale into 3 channels
6. apply normalization logic compatible with the training setup
7. add a batch dimension

---

## API Endpoints

### Health check

**GET /**

Returns a simple status response.

Example response:

```json
{
  "status": "ok",
  "message": "Welcome to the Motor Impairment Score API"
}
```

### Predict wave drawing

**POST /predict/wave**

Accepts a wave drawing image and returns a prediction.

### Predict spiral drawing

**POST /predict/spiral**

Accepts a spiral drawing image and returns a prediction.

### Input format

Both prediction endpoints expect:

- `multipart/form-data`
- a single file field named `file`

### Example response

```json
{
  "drawing_type": "wave",
  "raw_logit": 8.7809,
  "sigmoid_probability": 0.9998,
  "motor_impairment_score": 43.98,
  "severity_level": "Mild",
  "description": "Slight motor irregularities observed.",
  "is_parkinson": true
}
```

### Response fields

- `drawing_type` - either `wave` or `spiral`
- `raw_logit` - raw model output before sigmoid
- `sigmoid_probability` - probability converted from the logit
- `motor_impairment_score` - normalized score between 0 and 100
- `severity_level` - severity category
- `description` - human-readable interpretation
- `is_parkinson` - boolean indicator derived from the severity level

---

## Severity Levels

The API classifies the result into one of these labels:

- **Normal Pattern** - no motor impairment detected
- **Mild** - slight motor irregularities observed
- **Moderate** - noticeable motor impairment detected
- **High** - significant motor impairment observed
- **Severe** - strong Parkinsonian motor patterns detected

The exact score thresholds are defined in [services/predictor.py](services/predictor.py).

---

## Local Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If needed, also install the runtime packages used by the API:

```bash
pip install fastapi uvicorn python-multipart
```

### 3. Run the server

```bash
uvicorn main:app --reload
```

### 4. Open the documentation

Visit:

```text
http://127.0.0.1:8000/docs
```

This opens the interactive Swagger UI for testing the API.

---

## Example Requests

### cURL example

```bash
curl -X POST "http://127.0.0.1:8000/predict/wave" \
   -F "file=@test/wave.png"
```

### Spiral example

```bash
curl -X POST "http://127.0.0.1:8000/predict/spiral" \
   -F "file=@test/spiral.png"
```

### Python example

```python
import requests

url = "http://127.0.0.1:8000/predict/wave"

with open("test/wave.png", "rb") as f:
      files = {"file": f}
      response = requests.post(url, files=files)

print(response.json())
```

---


## Deployment

### Docker

The repository contains a Dockerfile that:

- installs system dependencies required by OpenCV
- installs the Python dependencies
- runs the API on port `7860`

### Hugging Face Spaces

The repository is set up for deployment as a Docker Space on Hugging Face.

Deployment flow:

1. Push changes to the `main` branch
2. GitHub Actions runs the workflow in [deploy.yml](.github/workflows/deploy.yml)
3. The workflow pushes the repository to the Hugging Face Space repository
4. Hugging Face rebuilds and redeploys the Space

Target Hugging Face repository:

`https://huggingface.co/spaces/xplorers/MIS_API`

### Required secret

The GitHub Actions workflow needs this secret:

- `HF_TOKEN` - Hugging Face write token

Make sure the token has permission to push to the target Space.

---

## Important Files

### [main.py](main.py)

Contains:

- the FastAPI app
- startup model loading
- CORS configuration
- image prediction endpoints

### [services/predictor.py](services/predictor.py)

Contains:

- Hugging Face model download logic
- image preprocessing
- wave and spiral prediction functions
- severity interpretation logic

### [Dockerfile](Dockerfile)

Contains:

- Python base image
- OpenCV system libraries
- application startup command

### [.github/workflows/deploy.yml](.github/workflows/deploy.yml)

Contains:

- GitHub Actions deployment logic
- authenticated push to Hugging Face Spaces

---

## License and Usage

No explicit license file is currently included in the repository.

If you plan to publish or share the project publicly, add a license file and review the Hugging Face model and deployment permissions.
