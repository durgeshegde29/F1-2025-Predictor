import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create raw rookie F2 data manually
data = {
    'Driver': [
        'Gabriel Bortoleto', 'Isack Hadjar', 'Oliver Bearman',
        'Andrea Kimi Antonelli', 'Jack Doohan'
    ],
    'FinalPosition': [1, 2, 12, 6, 5],
    'Points': [214.5, 192, 105, 150, 160],
    'Wins': [2, 4, 3, 2, 2],
    'Podiums': [8, 6, 3, 4, 5],
    'FastestLaps': [2, 2, 1, 1, 2],
    'DNFs': [1, 2, 3, 1, 2],
    'Races': [26, 26, 26, 26, 26]  # typical F2 season length
}

df = pd.DataFrame(data)

# Step 2: Feature engineering
df['AvgPoints'] = df['Points'] / df['Races']

# Step 3: Normalize relevant columns
features = ['FinalPosition', 'DNFs', 'AvgPoints', 'Wins', 'Podiums', 'FastestLaps']
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# Step 4: Compute StruggleScore (higher = more likely to struggle in F1)
df_scaled['StruggleScore'] = (
    0.3 * df_scaled['FinalPosition'] +
    0.25 * df_scaled['DNFs'] +
    0.2 * (1 - df_scaled['AvgPoints']) +
    0.1 * (1 - df_scaled['Wins']) +
    0.1 * (1 - df_scaled['Podiums']) +
    0.05 * (1 - df_scaled['FastestLaps'])
)

# Step 5: Save final output
df_final = df_scaled[['Driver', 'StruggleScore']]
df_final.to_csv('rookie_strugglescore_2025.csv', index=False)

print("✅ Rookie struggle scores saved to 'rookie_strugglescore_2025.csv'")
print(df_final)