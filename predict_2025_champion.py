import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load local CSVs (in the same folder)
df_2024 = pd.read_csv('driver_summary_2024.csv')
df_2025 = pd.read_csv('driver_start_form_2025.csv')

# Step 1: Aggregate 2025 driver data
agg_2025 = df_2025.groupby('Driver').agg({
    'Team': 'last',
    'Points': 'sum',
    'FinishPos': 'mean',
    'FastestLap': 'min',
    'AvgLapTime': 'mean',
    'SoftLaps': 'sum',
    'MediumLaps': 'sum',
    'HardLaps': 'sum'
}).reset_index()

agg_2025.columns = [
    'Driver', 'Team2025', 'TotalPoints_2025', 'AvgFinish_2025',
    'BestLap_2025', 'AvgLap_2025', 'Soft_2025', 'Medium_2025', 'Hard_2025'
]

# Step 2: Merge with 2024 summary
df_combined = pd.merge(df_2024, agg_2025, on='Driver', how='outer')

# Step 3: Fill missing values (rookies will have no 2024 data)
df_combined.fillna({
    'Team': df_combined['Team2025'],
    'Races': 0, 'AvgFinish': 20, 'Wins': 0, 'Podiums': 0,
    'BestLapTime': df_combined['BestLap_2025'],
    'AvgLapTime': df_combined['AvgLap_2025'],
    'TotalSoftLaps': 0, 'TotalMediumLaps': 0, 'TotalHardLaps': 0,
    'TotalPoints': 0, 'TotalDNFs': 0
}, inplace=True)

# Step 4: Normalize numeric features
features_to_scale = [
    'AvgFinish', 'Wins', 'Podiums', 'BestLapTime', 'AvgLapTime', 'TotalDNFs',
    'AvgFinish_2025', 'TotalPoints_2025', 'BestLap_2025', 'AvgLap_2025'
]

scaler = MinMaxScaler()
df_combined[features_to_scale] = scaler.fit_transform(df_combined[features_to_scale])

# Step 5: Scoring formula
df_combined['PredictionScore'] = (
    0.3 * (1 - df_combined['AvgFinish']) +
    0.2 * df_combined['Wins'] +
    0.15 * df_combined['Podiums'] +
    0.05 * (1 - df_combined['AvgLapTime']) +
    0.05 * (1 - df_combined['BestLapTime']) +
    0.05 * (1 - df_combined['TotalDNFs']) +
    0.1 * (1 - df_combined['AvgFinish_2025']) +
    0.05 * df_combined['TotalPoints_2025'] +
    0.025 * (1 - df_combined['AvgLap_2025']) +
    0.025 * (1 - df_combined['BestLap_2025'])
)

# Step 6: Sort and show predictions
df_combined_sorted = df_combined.sort_values(by='PredictionScore', ascending=False)
final_predictions = df_combined_sorted[['Driver', 'Team2025', 'PredictionScore']].reset_index(drop=True)

# Save to file
final_predictions.to_csv('final_predicted_2025_f1_ranking.csv', index=False)
print("✅ Final prediction saved to 'final_predicted_2025_f1_ranking.csv'")
print("\n🔮 Top 5 Predicted Drivers:\n")
print(final_predictions.head(5))