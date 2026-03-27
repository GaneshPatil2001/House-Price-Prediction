# 🏡 California House Price Prediction

This project focuses on predicting house prices using the California Housing dataset. It applies machine learning techniques to analyze key features such as income, house age, location, and population to estimate property prices.

---

## 📌 Project Overview

The goal of this project is to build an end-to-end machine learning pipeline that:

- Loads and processes housing data  
- Performs data preprocessing and feature engineering  
- Trains a regression model  
- Evaluates model performance  
- Deploys the model using a Flask web application  

---

## 🚀 Features

- Data preprocessing and scaling  
- Exploratory Data Analysis (EDA)  
- Model training using regression algorithms  
- Model serialization using Pickle  
- REST API creation using Flask  
- Simple UI using HTML  
- Deployment-ready structure (Render compatible)  

---

## 🛠️ Software & Tools Requirements

Make sure you have the following installed:

1. [GitHub Account](https://github.com)  
2. [Render Account](https://render.com)  
3. [VS Code IDE](https://code.visualstudio.com/)  
4. Python (>= 3.8)  
5. pip (Python package manager)  

---

## 📊 Dataset Information

The dataset includes the following features:

1. **MedInc** → Median Income  
2. **HouseAge** → Age of the house  
3. **AveRooms** → Average rooms  
4. **AveBedrms** → Average bedrooms  
5. **Population** → Population in area  
6. **AveOccup** → Average occupancy  
7. **Latitude** → Geographic coordinate  
8. **Longitude** → Geographic coordinate  

---

## 🎯 Target Variable

- **Price** → House price  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/GaneshPatil2001/House-Price-Prediction.git
cd California-House-Pricing
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

### 3️⃣ Activate Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5️⃣ Run the Application

```bash
python app.py
```

Open your browser and go to:  
👉 http://127.0.0.1:5000/

---

## 🌐 Deployment on Render

1. Push your project to GitHub  
2. Login to Render  
3. Click **New Web Service**  
4. Connect your GitHub repository  

### 🔧 Configuration

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
gunicorn app:app
```

5. Click **Deploy** 🎉  

---

## 📡 API Usage

### 🔹 Endpoint

```
POST /predict
```

---

### 🔹 Sample JSON Input

```json
{
  "MedInc": 5.0,
  "HouseAge": 20,
  "AveRooms": 6,
  "AveBedrms": 1,
  "Population": 1000,
  "AveOccup": 3,
  "Latitude": 34.05,
  "Longitude": -118.25
}
```

---

### 🔹 Sample Output

```json
{
  "predicted_price": 2.35
}
```

---

## 📈 Model Workflow

- Data Collection  
- Data Cleaning  
- Feature Scaling (StandardScaler)  
- Model Training (Linear Regression)  
- Model Evaluation  
- Model Saving using Pickle  
- Deployment using Flask  

---

## 📈 Future Improvements

- Add advanced ML models (Random Forest, XGBoost)  
- Improve UI with interactive dashboard  
- Add model evaluation metrics display  
- Add Docker support  
- Implement CI/CD pipeline  

---

## 🤝 Contributing

Contributions are welcome!

### Steps:

1. Fork the repository  
2. Create a new branch  
3. Make your changes  
4. Commit changes  
5. Push to branch  
6. Create Pull Request  

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Ganesh Patil**  
*(Data Scientist)*  

- GitHub: https://github.com/GaneshPatil2001  
- LinkedIn: https://www.linkedin.com/in/ganeshpatil2001  

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!