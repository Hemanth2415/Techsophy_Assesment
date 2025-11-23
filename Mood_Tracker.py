import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


df = pd.read_excel('/content/Daylio_Abid.csv.xlsx')

df['date'] = pd.to_datetime(df['full_date'], format='%d/%m/%Y')

df['mood_clean'] = df['mood'].str.strip().str.lower()

df.head()



mood_to_score = {
    'over the moon': 5,
    'happiest day': 5,
    'excited': 5,
    'blessed': 5,
    'yolo': 5,
    'cool': 5,
    'chill': 5,
    'good': 5,
    'focused': 4,
    'wondering': 3,
    'confused': 3,
    'meh': 3,
    'hungry': 3,
    'weak': 2,
    'worried': 2,
    'scared': 2,
    'angry': 1,
    'bad': 1,
    'sad af': 1,
    'awful': 1,
    'triggered': 1,
    'sick': 1,
}

df['mood_score'] = df['mood_clean'].map(mood_to_score)

unmapped = df.loc[df['mood_score'].isna(), 'mood_clean'].dropna().unique()

if len(unmapped) > 0:
    print(f"Unmapped moods ({len(unmapped)}):")
    print(unmapped)
else:
    print("All moods are mapped.")

df['mood_score'] = df['mood_score'].fillna(3).astype(float)


daily = (
    df.groupby('date', as_index=False)
      .agg(
          mood_score=('mood_score', 'mean'),
          weekday=('weekday', 'first'),
          activities=('activities', lambda x: ' | '.join(str(v) for v in x.dropna()))
      )
      .sort_values('date')
)

full_range = pd.date_range(start=daily['date'].min(),
                           end=daily['date'].max(),
                           freq='D')

daily = (
    daily.set_index('date')
         .reindex(full_range)
         .rename_axis('date')
)

daily['mood_score'] = daily['mood_score'].interpolate(limit_direction='both')
daily['weekday'] = daily['weekday'].ffill().bfill()
daily['activities'] = daily['activities'].fillna('no entry')

daily.head()


daily['day_of_week'] = daily.index.dayofweek
daily['is_weekend'] = (daily['day_of_week'] >= 5).astype(int)

rolling_windows = {
    'mood_3d_avg': 3,
    'mood_7d_avg': 7,
    'mood_30d_avg': 30
}

for col, window in rolling_windows.items():
    daily[col] = daily['mood_score'].rolling(window=window, min_periods=1).mean()

daily['mood_7d_std'] = (
    daily['mood_score']
    .rolling(window=7, min_periods=1)
    .std()
    .fillna(0)
)



activity_series = (
    daily['activities']
    .fillna('')
    .astype(str)
    .str.lower()
    .str.replace(' ', '', regex=False)
    .str.split('|')
)

from collections import Counter

activity_counter = Counter([a for acts in activity_series for a in acts if a])
top_activities = [a for a, _ in activity_counter.most_common(10)]

print("Top activities:", top_activities)

for act in top_activities:
    daily[f'act_{act}'] = activity_series.apply(lambda acts: int(act in acts))



fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(daily.index, daily['mood_score'], marker='.', alpha=0.6, label='Daily Mood')
ax.plot(daily.index, daily['mood_7d_avg'], linewidth=2, label='7-Day Average')
ax.plot(daily.index, daily['mood_30d_avg'], linewidth=2, label='30-Day Average')

ax.set(
    title='Mood Trend Over Time',
    xlabel='Date',
    ylabel='Mood Score (1–5)'
)

ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



weekday_mood = daily.groupby('day_of_week')['mood_score'].mean()

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(weekday_mood.index, weekday_mood.values)

ax.set(
    title='Average Mood by Weekday',
    xlabel='Day of Week',
    ylabel='Average Mood Score (1–5)'
)

ax.set_xticks(range(7))
ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

plt.tight_layout()
plt.show()



activity_cols = [c for c in daily.columns if c.startswith('act_')]

activity_mood = (
    daily[activity_cols]
    .mul(daily['mood_score'], axis=0)
    .replace(0, np.nan)
    .mean()
    .sort_values(ascending=False)
)

activity_mood



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features_for_cluster = ['mood_score', 'mood_7d_avg', 'mood_7d_std', 'is_weekend'] + activity_cols

X_cluster = daily[features_for_cluster].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
daily['mood_cluster'] = kmeans.fit_predict(X_scaled)

daily[['mood_score', 'mood_7d_avg', 'mood_cluster']].head()



features_for_anom = ['mood_score', 'mood_3d_avg', 'mood_7d_avg', 'mood_7d_std', 'is_weekend']

X_anom = daily[features_for_anom].fillna(0)

scaler_anom = StandardScaler()
X_anom_scaled = scaler_anom.fit_transform(X_anom)

iso = IsolationForest(contamination=0.1, random_state=42)
daily['is_anomaly'] = (iso.fit_predict(X_anom_scaled) == -1).astype(int)


fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(daily.index, daily['mood_score'], marker='.', label='Mood')
anoms = daily[daily['is_anomaly'] == 1]
ax.scatter(anoms.index, anoms['mood_score'], marker='x', s=80, label='Anomaly (Unusual Day)')

ax.set(
    xlabel='Date',
    ylabel='Mood Score',
    title='Mood with Detected Anomalies'
)

ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


LOW_MOOD_THRESHOLD = 2.0

daily['is_low_mood'] = (daily['mood_score'] <= LOW_MOOD_THRESHOLD).astype(int)
daily['low_mood_streak'] = (
    daily['is_low_mood']
    .groupby((daily['is_low_mood'] != daily['is_low_mood'].shift()).cumsum())
    .cumsum()
    * daily['is_low_mood']
)


def classify_severity(row):
    if row['low_mood_streak'] >= 7 or (row['is_anomaly'] == 1 and row['mood_score'] <= 2):
        return "Critical – please take care and consider reaching out for support"
    elif row['mood_score'] <= 2 or row['is_anomaly'] == 1:
        return "Concerning – reflect on your well-being and try supportive activities"
    elif row['mood_score'] < row['mood_7d_avg']:
        return "Moderate – slight drop in mood compared to recent days"
    else:
        return "Stable – mood appears consistent or improving"

daily['wellbeing_status'] = daily.apply(classify_severity, axis=1)


print(daily[['mood_score', 'mood_7d_avg', 'is_anomaly', 'low_mood_streak', 'wellbeing_status', 'recommendation']])
