import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
print("Running from:", os.getcwd())
print("Script location:", os.path.dirname(os.path.abspath(__file__)))
print("Data dir:", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
print("Data folder exists:", os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')))



script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

players = pd.read_csv(os.path.join(data_dir, 'cleaned_players.csv'))

players = players[players['overall'] >= 75]
print(f"Filtered player pool: {len(players)}")

def generate_random_team(df, n=11):
    return df.sample(n)

def team_strength(team_df):
    return (
        team_df['overall'].mean() * 0.30 +
        team_df['potential'].mean() * 0.05+
        team_df['value_eur'].mean() * 0.05 +
        team_df['international_reputation'].mean() * 0.10 +
        team_df['pace'].mean() * 0.10 +
        team_df['shooting'].mean() * 0.1225 +
        team_df['passing'].mean() * 0.1225 +
        team_df['defending'].mean() * 0.10 +
        team_df['physic'].mean() * 0.10
    )

training_data = []
for i in range(10000):
    team = generate_random_team(players)
    row = {
        'avg_overall': team['overall'].mean(),
        'avg_potential': team['potential'].mean(),
        'avg_value_eur': team['value_eur'].mean(),
        'avg_international_reputation': team['international_reputation'].mean(),    
        'avg_pace': team['pace'].mean(),
        'avg_shooting': team['shooting'].mean(),
        'avg_passing': team['passing'].mean(),
        'avg_defending': team['defending'].mean(),
        'avg_physic': team['physic'].mean(),
        'std_overall': team['overall'].std(),
        'min_overall': team['overall'].min(),
        'max_overall': team['overall'].max(),
        'strength': team_strength(team)
    }
    training_data.append(row)

    # Print progress every 1000 teams
    if (i + 1) % 1000 == 0:
        print(f"Generated {i + 1} teams...")

print("Done!")

df = pd.DataFrame(training_data)

df.to_csv(os.path.join(data_dir, 'processed_teams.csv'), index=False)
print("Saved processed_teams.csv")
