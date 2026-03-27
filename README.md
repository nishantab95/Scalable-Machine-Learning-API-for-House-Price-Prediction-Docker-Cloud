# 🏠 AI-Powered House Price Prediction System

### End-to-End Machine Learning Project with Deployment on Google Cloud Run
---

## 📌 Project Overview
This project is an **end-to-end Machine Learning system** that predicts house prices in Bangalore using advanced regression models. It covers the full ML lifecycle — from data preprocessing and model building to API development, containerization, and cloud deployment.
The system is designed to simulate a **real-world production ML pipeline**, making it suitable as a capstone project for Data Science and Machine Learning roles.

---

## 🎯 Problem Statement
Real estate price prediction is a complex task due to multiple influencing factors such as location, size, number of bedrooms, and amenities.

👉 Goal:
Build a robust ML model to accurately predict house prices based on structured input features.
---

## 🧠 Machine Learning Approach

### Models Used:
* Gradient Boosting Regressor
* XGBoost Regressor

### Technique:
* Model stacking (ensemble learning)
* Feature engineering and preprocessing
* Performance optimization using evaluation metrics
---

## 📊 Dataset
* Cleaned Bangalore housing dataset
* Includes features such as:

  * Area (sqft)
  * Number of bedrooms (BHK)
  * Bathrooms
  * Location encoding
  * Additional engineered features
---

## ⚙️ Tech Stack
| Category        | Tools & Technologies  |
| --------------- | --------------------- |
| Programming     | Python                |
| ML Libraries    | scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy         |
| Backend         | Flask                 |
| API             | REST API              |
| Deployment      | Docker                |
| Cloud Platform  | Google Cloud Run      |
| Version Control | Git, GitHub           |
---

## 🏗️ Project Architecture
```text
User Input → Flask API → ML Model → Prediction → Response
```
---

## 🚀 Features
* ✅ End-to-end ML pipeline
* ✅ REST API for predictions
* ✅ Dockerized application
* ✅ Deployed on Google Cloud Run
* ✅ Scalable and production-ready setup
* ✅ Modular project structure
---

## 🌍 Live Demo
🔗 **Deployed API Endpoint:**
```
https://your-cloud-run-url.run.app
```
---

## 🔌 API Usage

### Endpoint:
```
POST /predict
```

### Request:
```json
{
  "data": [1000, 2, 2, 2, 2, 2, 2, 2, 4]
}
```

### Response:

```json
{
  "prediction": 8500000
}
```
---

## 🧪 Model Performance
| Metric   | Value            |
| -------- | ---------------- |
| R² Score | (Add your value) |
| MAE      | (Add your value) |
| RMSE     | (Add your value) |
---

## 🐳 Docker Deployment

### Build:
```
docker build -t house-price-app .
```

### Run:
```
docker run -p 8080:8080 -e PORT=8080 house-price-app
```
---

## ☁️ Cloud Deployment
The application is deployed using:
* Docker containerization
* Google Cloud Run (serverless deployment)
This ensures:
* Auto-scaling
* High availability
* Zero server management
---

## 📂 Project Structure
```text
├── app/
│   ├── app.py
│   └── templates/
│       └── home.html
│
├── model/
│   └── xgb_gbr_stacked_model.pkl
│
├── data/
│   └── clean_data.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── training.ipynb
│
├── Dockerfile
├── requirements.txt
├── README.md
```
---

## 🔐 Security & Deployment Notes
* Public API enabled for demonstration purposes
* Can be secured using authentication or API Gateway
* Binary authorization disabled (not required for small projects)
---

## 📈 Future Improvements
* Add frontend UI (Streamlit / React)
* Improve feature engineering
* Hyperparameter tuning
* Integrate real-time data
* Add monitoring & logging (ML Ops)
---

## 💼 Key Skills Demonstrated
* Machine Learning Model Development
* Feature Engineering
* Model Deployment (MLOps basics)
* REST API Development
* Docker & Containerization
* Cloud Deployment (Google Cloud Run)
* Debugging Real-World Issues
---

## 🧠 Learning Outcomes
* Handling dependency and version conflicts
* Building production-ready ML systems
* Deploying scalable APIs
* Structuring industry-level ML projects
---

## 👨‍💻 Author
**Nishant Bilagi**
---
## ⭐ If you found this project useful, consider giving it a star!
