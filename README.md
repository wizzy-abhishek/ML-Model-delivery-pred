# 📦 Delivery Date Prediction — ML Microservice

A machine learning microservice that predicts **when a product will be delivered**, built for e-commerce platforms like Amazon, Flipkart, and Myntra. Instead of returning a single day, the model returns a **range of probable delivery days with confidence percentages** — giving users a more honest and intuitive delivery estimate.

---

## 🧠 Problem Statement

Predicting delivery dates is a **multiclass classification problem**. Given details about a shipment — pickup zone, delivery zone, product type, weight, and weather — the model predicts the probability of delivery on each possible day, returning the top 3 most likely delivery windows.

---

## 🚀 Features

- **Probability-based prediction** — e.g. 60% Day 3, 21% Day 2, 17% Day 4
- **Top 3 delivery window** returned instead of a single hard prediction
- **India-aware seasonal weather** — automatically infers weather from the current month since no live weather API is used
- **REST API via Flask** — designed to plug into Spring Boot or any backend

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Model | XGBoost (`multi:softprob`) |
| Encoding | Scikit-learn `ColumnTransformer` |
| API | Flask |
| Serialization | Joblib |
| Language | Python 3.13 |

---

## 📁 Project Structure

```
Delivery_prediction/
├── app.py                    # Flask API
├── model_training.ipynb      # Jupyter notebook — EDA, training, evaluation
├── xgb_model.pkl             # Saved XGBoost model
├── column_transformer.pkl    # Saved encoder (must be used with model)
├── delivery_processed1.csv   # Training dataset
└── requirements.txt          # Dependencies
```

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `pickup_prefix` | First 2 digits of pickup pincode — represents source zone |
| `delivery_prefix` | First 2 digits of delivery pincode — represents destination zone |
| `product_category` | STANDARD or DELICATE |
| `weight_bucket` | Weight binned into categories (tiny, light, medium, heavy, bulk) |
| `weather` | Auto-inferred from India's seasonal patterns |

> **Note:** Full pincodes were dropped in favour of prefixes to improve model generalization across unseen pincode combinations.

---

## 🌦️ Weather Inference (India Seasonal Pattern)

Since no live weather API is integrated, weather is inferred from the current month:

| Months | Weather | Season |
|---|---|---|
| Jan – Apr | SUNNY | Winter → Summer |
| May | STORMY | Pre-monsoon |
| Jun – Sep | RAINY | Monsoon |
| Oct – Nov | CLOUDY | Post-monsoon |
| Dec | SUNNY | Winter |

---

## 📡 API Usage

### Start the server
```bash
source venv/bin/activate
python app.py
```

### POST `/predict`

**Request**
```json
{
  "pickup_prefix": 20,
  "delivery_prefix": 81,
  "product_category": "DELICATE",
  "weight_bucket": "Medium Parcel"
}
```

**Response**
```json
{
  "predicted_day": 3,
  "weather_used": "SUNNY",
  "top_3_days": [
    { "day": 3, "probability": 60.0 },
    { "day": 2, "probability": 21.4 },
    { "day": 4, "probability": 17.2 }
  ]
}
```

### GET `/health`
```json
{ "status": "ok", "weather_this_month": "SUNNY" }
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Accuracy | ~56% |
| Problem Type | Multiclass Classification (10 classes) |
| Evaluation | Accuracy + F1-score per class |

> The model performs well on human-validated test cases. Accuracy is currently under active improvement — the primary bottleneck is the absence of real-time weather data and the high cardinality of pincode-based routing.

---

## 🔭 Future Improvements

- [ ] Integrate live weather API (e.g. OpenWeatherMap) per delivery zone
- [ ] Add route feature (`pickup_prefix_delivery_prefix`) to capture zone-to-zone patterns
- [ ] Hyperparameter tuning via GridSearchCV or Optuna
- [ ] Experiment with LightGBM or CatBoost for comparison
- [ ] Collect more training data for rare delivery day classes

---

## ⚙️ Installation

```bash
git clone <repo-url>
cd Delivery_prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

NOTE: README.md was made by AI, if there is any mistake init or in my project kindly let me know.
