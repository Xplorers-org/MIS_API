---
title: Parkinson's Motor Impairment Predictor
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 🧠 Parkinson's Motor Impairment Score API

 **FastAPI** backend that evaluates patient drawings (wave and spiral patterns) to detect and quantify motor impairments associated with Parkinson's disease.

This service utilizes deep learning models (**VGG19** and **ResNet101**) automatically dynamically fetched from Hugging Face along with a custom OpenCV pre-processing pipeline to score severity precisely.

---

## 🚀 Features
* **Lightning Fast:** Implements a FastAPI lifecycle manager to preload multi-hundred-megabyte machine learning models straight into RAM on server startup, ensuring 0-second prediction latency.
* **Dual Classifiers:**
  * **Wave Patterns:** Analyzed via a VGG19 backbone.
  * **Spiral Patterns:** Analyzed via a ResNet101 backbone.
* **Torch-less:** Fully replicates the original PyTorch training `torchvision` image transformation pipeline using strictly `numpy` and `cv2` for ultra-lightweight deployment.

---

## 🛠️ API Endpoints

### 1. `POST /predict/wave`
Accepts a drawn "wave" pattern image overlay and calculates severity.

### 2. `POST /predict/spiral`
Accepts a drawn "spiral" pattern image overlay and calculates severity.

**Input Format:** `multipart/form-data` with an image file.

**Response Example:**
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

---

## 💻 Local Development

1. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install fastapi uvicorn python-multipart
   ```

3. **Run the Server:**
   ```bash
   uvicorn main:app --reload
   ```

4. Go to `http://127.0.0.1:8000/docs` to test the API directly using Swagger UI!

---

## ☁️ Hugging Face Deployment (Docker)
This repository is pre-configured to automatically deploy as a **Docker Space** to Hugging Face via GitHub Actions.

1. Any changes pushed to the `main` branch trigger `.github/workflows/deploy.yml`
2. GitHub forcefully mirrors this repository to Hugging Face Hub securely using an injected `HF_TOKEN`.
3. Hugging Face reads the attached `Dockerfile` and dynamically exposes the FastAPI service on Port `7860`.
