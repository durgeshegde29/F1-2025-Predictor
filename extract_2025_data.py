import os
import fastf1
import pandas as pd

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache_folder')

# Define 2025 rounds (first 3)
rounds = [1, 2, 3]
sessions = ['R', 'Sprint', 'Sprint Shootout']

all_data = []

for rnd in rounds:
    for session_type in sessions:
        try:
            session = fastf1.get_session(2025, rnd, session_type)
            session.load()
            print(f"✅ Loaded: Round {rnd} - {session.event['EventName']} - {session_type}")

            results = session.results
            laps = session.laps

            for index, row in results.iterrows():
                drv = row['Abbreviation']
                team = row['TeamName']
                pos = row['Position']
                points = row['Points'] if 'Points' in row else 0
                status = row['Status']

                # Lap stats
                driver_laps = laps.pick_driver(drv)
                if not driver_laps.empty:
                    fastest = driver_laps['LapTime'].min()
                    avg_lap = driver_laps['LapTime'].mean()
                    compound_counts = driver_laps['Compound'].value_counts().to_dict()
                    soft = compound_counts.get('SOFT', 0)
                    medium = compound_counts.get('MEDIUM', 0)
                    hard = compound_counts.get('HARD', 0)
                else:
                    fastest = avg_lap = pd.NaT
                    soft = medium = hard = 0

                all_data.append({
                    'Round': rnd,
                    'Session': session_type,
                    'Driver': drv,
                    'Team': team,
                    'FinishPos': pos,
                    'Points': points,
                    'Status': status,
                    'FastestLap': fastest.total_seconds() if pd.notna(fastest) else None,
                    'AvgLapTime': avg_lap.total_seconds() if pd.notna(avg_lap) else None,
                    'SoftLaps': soft,
                    'MediumLaps': medium,
                    'HardLaps': hard
                })

        except Exception as e:
            print(f"❌ Skipped: Round {rnd} - {session_type}: {e}")

# Convert to DataFrame and save
df = pd.DataFrame(all_data)
df.to_csv('driver_start_form_2025.csv', index=False)
print("\n📁 Saved driver early 2025 performance to 'driver_start_form_2025.csv'")