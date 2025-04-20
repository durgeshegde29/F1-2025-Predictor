import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# === Step 1: Load CSVs ===
df_2024 = pd.read_csv('driver_summary_2024.csv')
df_2025 = pd.read_csv('driver_start_form_2025.csv')
rookies = pd.read_csv('rookie_strugglescore_2025.csv')

# === Step 2: Aggregate early 2025 F1 performance ===
df_agg_2025 = df_2025.groupby('Driver').agg({
    'Team': 'last',
    'Points': 'sum',
    'FinishPos': 'mean',
    'FastestLap': 'mean',
    'AvgLapTime': 'mean'
}).reset_index()
df_agg_2025.columns = ['Driver', 'Team2025', 'TotalPoints_2025', 'AvgFinish_2025', 'BestLap_2025', 'AvgLap_2025']

# === Step 3: Extract 2024 features for returnees ===
df_2024_features = df_2024[['Driver', 'AvgFinish', 'Wins', 'TotalDNFs', 'TotalPoints', 'AvgLapTime', 'BestLapTime']]

# === Step 4: Merge everything ===
merged = pd.merge(df_agg_2025, df_2024_features, on='Driver', how='left')
merged = pd.merge(merged, rookies, on='Driver', how='left')

# === Step 5: Fill missing values ===
merged['RookieStruggleScore'] = merged['StruggleScore'].fillna(0)
merged.drop(columns=['StruggleScore'], inplace=True)
merged.fillna({
    'AvgFinish': 20,
    'Wins': 0,
    'TotalDNFs': 3,
    'TotalPoints': 0,
    'AvgLapTime': merged['AvgLap_2025'],
    'BestLapTime': merged['BestLap_2025']
}, inplace=True)

# === Step 6: Label bottom 5 based on 2024 points ===
bottom_5_2024 = df_2024.sort_values(by='TotalPoints').head(5)['Driver'].tolist()
merged['isBottom5'] = merged['Driver'].isin(bottom_5_2024).astype(int)

# === Step 7: Feature setup ===
features = [
    'AvgFinish', 'Wins', 'TotalDNFs', 'TotalPoints',
    'AvgLapTime', 'BestLapTime', 'AvgFinish_2025',
    'TotalPoints_2025', 'AvgLap_2025', 'BestLap_2025',
    'RookieStruggleScore', 'Team2025'
]
target = 'isBottom5'

X = merged[features]
y = merged[target]

# === Step 8: Preprocessing ===
numeric_features = [f for f in features if f != 'Team2025']
categorical_features = ['Team2025']

preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# === Step 9: Model pipeline ===
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# === Step 10: Train model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# === Step 11: Predict all 2025 drivers with probabilities ===
probs = pipeline.predict_proba(X)[:, 1]  # Probability of being in bottom 5
merged['Bottom5Prob'] = probs

# === Step 12: Sort and select top 5 most likely bottom finishers ===
bottom5_df = merged.sort_values(by='Bottom5Prob', ascending=False).head(5)
bottom5_df = bottom5_df[['Driver', 'Team2025', 'Bottom5Prob']]
bottom5_df.to_csv('predicted_bottom5_2025.csv', index=False)

print("\n💀 Predicted Bottom 5 Drivers of 2025 F1 Season (Based on Probability):")
print(bottom5_df)