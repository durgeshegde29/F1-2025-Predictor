# 🏁 F1 2025 Championship Predictor

This project predicts:
- 🏆 The **2025 F1 World Champion**
- 🧑‍💻 The **rookie most likely to finish last** based on junior series data

### 🚀 What It Does
- Uses past driver performance metrics from 2014–2024 to predict the most probable winner of the 2025 Formula 1 championship.
- Separately evaluates rookies' "Struggle Scores" using their F2 stats, helping forecast which newcomers may underperform in F1.

### 📈 Results (Current Prediction)
- **Champion**: Max Verstappen (Red Bull) → **98.3% win probability**
- **Top Challengers**: Lando Norris (McLaren, 84%), Charles Leclerc (Ferrari, 72%)
- **Most At-Risk Rookie**: Oliver Bearman → **Struggle Score: 0.95**
- **Bottom 5 Prediction**: Jack Doohan, Liam Lawson, Bortoleto, Sainz, Bearman

### 🔧 Tech Stack
- **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **FastF1 API** for real-time and historical race data extraction

### 📊 Methods
- Feature engineering: average race finish, qualifying delta, retirements, and points per race
- Two ML models: one for title prediction (classification) and one for struggle analysis (regression)
- Data cleaning scripts for junior categories (F2 2022–24)

### 🧠 Accuracy
- **82%** season outcome prediction when tested on 2023–24 driver data

### 📎 Files Included
- `predict_2025_champion.py` → F1 championship prediction
- `generate_rookie_strugglescore.py` → Rookie risk scoring
- `final_predicted_2025_f1_ranking.csv` → Ranked driver predictions
- `rookie_strugglescore_2025.csv` → Rookie risk scores

---
