import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# this line creates df - was missing from your file
df = pd.read_csv(os.path.join(data_dir, 'processed_teams.csv'))

print(f"Training data shape: {df.shape}")

features = ['avg_overall', 'avg_pace', 'avg_shooting', 
            'avg_passing', 'avg_defending', 'avg_physic',
            'std_overall', 'min_overall', 'max_overall']

X = df[features]
y = df['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"R² Score: {model.score(X_test, y_test):.4f}")
print(f"MAE: {mean_absolute_error(y_test, model.predict(X_test)):.4f}")

joblib.dump(model, 'team_strength_model.pkl')
