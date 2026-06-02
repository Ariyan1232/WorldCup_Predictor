import pandas as pd

players = pd.read_csv('male_players.csv')
teams = pd.read_csv('male_teams.csv')
results = pd.read_csv('results.csv')
shootouts = pd.read_csv('shootouts.csv')

wc_matches = results[results['tournament'] == 'FIFA World Cup']

print(players.groupby('nationality_name').size().sort_values(ascending=False).head(20))