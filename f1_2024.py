import os
import fastf1
import pandas as pd
import numpy as np

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache_folder')

# FIA Points system
points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Store all race-level data per driver
all_race_data = []

# Loop through all 24 rounds of the 2024 season
for rnd in range(1, 25):
    try:
        session = fastf1.get_session(2024, rnd, 'R')
        session.load()
        race_name = session.event['EventName']
        print(f"✅ Loaded: Round {rnd} - {race_name}")

        results = session.results
        laps = session.laps

        for index, row in results.iterrows():
            drv = row['Abbreviation']
            team = row['TeamName']
            finish_pos = row['Position']
            dnf = 1 if row['Status'] != 'Finished' else 0
            points = points_map.get(finish_pos, 0)

            # Get lap data for driver (if available)
            driver_laps = laps.pick_driver(drv)
            if not driver_laps.empty:
                fastest_lap = driver_laps['LapTime'].min()
                avg_lap = driver_laps['LapTime'].mean()
                compound_counts = driver_laps['Compound'].value_counts().to_dict()
                soft = compound_counts.get('SOFT', 0)
                medium = compound_counts.get('MEDIUM', 0)
                hard = compound_counts.get('HARD', 0)
            else:
                fastest_lap = pd.NaT
                avg_lap = pd.NaT
                soft = medium = hard = 0

            all_race_data.append({
                'Driver': drv,
                'Team': team,
                'Round': rnd,
                'Race': race_name,
                'FinishPos': finish_pos,
                'FastestLap': fastest_lap.total_seconds() if pd.notna(fastest_lap) else None,
                'AvgLapTime': avg_lap.total_seconds() if pd.notna(avg_lap) else None,
                'SoftLaps': soft,
                'MediumLaps': medium,
                'HardLaps': hard,
                'Points': points,
                'DNF': dnf
            })

    except Exception as e:
        print(f"❌ Error in Round {rnd}: {e}")

# Convert to DataFrame
df = pd.DataFrame(all_race_data)

# Group and aggregate per driver
summary = df.groupby('Driver').agg({
    'Team': 'last',
    'FinishPos': ['count', 'mean', lambda x: (x == 1).sum(), lambda x: (x <= 3).sum()],
    'FastestLap': 'min',
    'AvgLapTime': 'mean',
    'SoftLaps': 'sum',
    'MediumLaps': 'sum',
    'HardLaps': 'sum',
    'Points': 'sum',
    'DNF': 'sum'
}).reset_index()

# Rename columns
summary.columns = [
    'Driver', 'Team', 'Races', 'AvgFinish', 'Wins', 'Podiums',
    'BestLapTime', 'AvgLapTime', 'TotalSoftLaps', 'TotalMediumLaps',
    'TotalHardLaps', 'TotalPoints', 'TotalDNFs'
]

# Save to CSV
summary.to_csv('driver_summary_2024.csv', index=False)
print("\n✅ Saved complete driver summary to 'driver_summary_2024.csv'")