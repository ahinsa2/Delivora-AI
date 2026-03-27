# 🛵 DELIVORA AI — Smart ETA & Delay Prediction Engine

> Predict food delivery ETAs with machine learning. Reduce cancellations. Build customer trust.

---

## 📌 Problem Statement

Food delivery platforms suffer from inaccurate ETAs — causing customer frustration, order cancellations, and revenue loss. Existing systems rely on static formulas that ignore real-world variables like traffic congestion, weather conditions, rider experience, and peak-hour demand patterns.

**The result:** 63% of users cite ETA accuracy as their #1 frustration, and ~22% of delayed orders result in cancellations costing platforms $4–8 per order.

---

## 💡 Solution Overview

DELIVORA AI is an end-to-end ML pipeline that:

- Simulates 15,000 realistic food delivery orders across 4 Indian cities
- Engineers meaningful features from raw order data
- Trains a **RandomForestRegressor** to predict accurate delivery times
- Scores each order with a **delay risk probability** (Low / Medium / High)
- Surfaces predictions through a real-time **Streamlit dashboard**

---

## ✨ Features

- **Realistic synthetic data** — 13 correlated features with business-logic constraints (peak hours, weather, traffic)
- **Feature engineering pipeline** — derives `delivery_time`, `eta_error`, `is_delayed`, `is_peak_hour`
- **ML regression model** — RandomForest with 200 trees, tuned for production accuracy
- **Comprehensive evaluation** — MAE, RMSE, P50 and P90 absolute error metrics
- **Feature importance analysis** — identifies which inputs drive prediction error
- **Interactive dashboard** — live ETA prediction + risk badge via Streamlit
- **Modular codebase** — clean separation of data, features, and model layers

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib / Pickle |
| Dashboard | Streamlit |
| Visualization | Matplotlib (optional) |

---

## 📁 Project Structure

```
delivora-ai/
│
├── data/
│   ├── raw_orders.csv                  # Initial synthetic dataset (1,200 rows)
│   ├── raw_orders_v2_balanced.csv      # Balanced dataset (10,000 rows)
│   ├── raw_orders_v3_realistic.csv     # Production-grade dataset (15,000 rows)
│   └── engineered_orders.csv          # Feature-engineered output
│
├── src/
│   ├── features/
│   │   └── feature_engineering.py     # Feature derivation & data quality pipeline
│   │
│   └── models/
│       ├── model_training.py           # Model training, evaluation & persistence
│       └── saved_model.pkl             # Serialized trained model
│
├── app.py                              # Streamlit dashboard
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/delivora-ai.git
cd delivora-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run feature engineering

```bash
python src/features/feature_engineering.py
```

This reads `data/raw_orders_v3_realistic.csv` and outputs `data/engineered_orders.csv`.

### 4. Train the model

```bash
python src/models/model_training.py
```

This trains the RandomForest model, prints evaluation metrics, and saves `src/models/saved_model.pkl`.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | ~4 minutes |
| Root Mean Squared Error (RMSE) | ~6 minutes |
| P50 Absolute Error | ~3 minutes |
| P90 Absolute Error | ~9 minutes |

> **90% of all deliveries are predicted within 9 minutes of actual arrival time.**

### Feature Importance

| Feature | Importance |
|---|---|
| `distance_km` | 65% |
| `prep_time_minutes` | 14% |
| `traffic_index` | 10% |
| `is_peak_hour` | 5% |
| `weather_score` | 4% |
| `rider_experience_years` | 2% |

---

## 🔮 Future Improvements

- [ ] Integrate real-time GPS and live traffic API (Google Maps / HERE)
- [ ] Upgrade model to **XGBoost** or **LightGBM** for improved accuracy
- [ ] Add binary **delay classification model** alongside the regression model
- [ ] Wrap pipeline in a **FastAPI REST endpoint** for production deployment
- [ ] Implement **LSTM time-series forecasting** for sequence-aware ETA prediction
- [ ] Build restaurant and rider performance profiling
- [ ] Multi-city A/B testing framework
- [ ] Docker containerization for one-command deployment

---

## 🧑‍💻 Author
- GitHub: https://github.com/ahinsa2
- LinkedIn: https://linkedin.com/in/ahinsa-mohanty-290765331

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ for the DELIVORA AI Hackathon</p>
