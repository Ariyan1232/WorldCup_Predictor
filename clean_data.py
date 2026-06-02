import pandas as pd
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')


cols = ['short_name', 'nationality_name', 'overall','potential','value_eur','international_reputation','pace', 'shooting', 'passing', 'defending', 'physic','player_positions']

players = pd.read_csv(os.path.join(data_dir, 'male_players.csv'), low_memory=False)

# Clean Up data file
players = players[cols].dropna()
players = players[players['overall'] > 0]
players = players.groupby('nationality_name').filter(lambda x: len(x) >= 10)

# Reduce Duplicates of Data
players = players.sort_values('overall', ascending=False)
players = players.drop_duplicates(subset='short_name', keep='first')
print(f"After deduplication: {len(players)}")

print(f"Total players loaded: {len(players)}")
print(f"Nationalities retained: {players['nationality_name'].nunique()}")
print(players.head())

# Save
players.to_csv(os.path.join(data_dir, 'cleaned_players.csv'), index=False)
print("Saved cleaned_players.csv")

