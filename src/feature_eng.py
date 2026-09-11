from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_json(BASE_DIR / "data" / "raw" / "weather.json")

# Convert Date/Time
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='mixed')

# Sort chronologically
df = df.sort_values('Date/Time').reset_index(drop=True)

# Keep required columns
df = df[['Date/Time', 'Temp_C', 'Rel Hum_%', 'Press_kPa']].copy()

# -----------------------------
# Time-based features
# -----------------------------
df['Hour'] = df['Date/Time'].dt.hour
df['Day'] = df['Date/Time'].dt.day
df['Month'] = df['Date/Time'].dt.month
df['DayOfWeek'] = df['Date/Time'].dt.dayofweek
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# -----------------------------
# Change-based features
# -----------------------------
df['Temp_Change'] = df['Temp_C'].diff()
df['Humidity_Change'] = df['Rel Hum_%'].diff()
df['Pressure_Change'] = df['Press_kPa'].diff()

# -----------------------------
# Rolling mean features
# -----------------------------
window = 10

df['Temp_Rolling_Mean'] = df['Temp_C'].rolling(window).mean()
df['Humidity_Rolling_Mean'] = df['Rel Hum_%'].rolling(window).mean()
df['Pressure_Rolling_Mean'] = df['Press_kPa'].rolling(window).mean()

# -----------------------------
# Rolling standard deviation
# -----------------------------
df['Temp_Rolling_Std'] = df['Temp_C'].rolling(window).std()
df['Humidity_Rolling_Std'] = df['Rel Hum_%'].rolling(window).std()
df['Pressure_Rolling_Std'] = df['Press_kPa'].rolling(window).std()

# Remove rows with NaN values caused by diff/rolling calculations
df = df.dropna().reset_index(drop=True)

# Check final dataset
print(df.head())
print(df.info())
print(df.shape)