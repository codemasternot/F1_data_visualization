#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


drivers = pd.read_csv('f1\\drivers.csv')
results = pd.read_csv('f1\\results.csv')


# In[3]:


merged = results.merge(drivers, left_on='driverId', right_on='driverId')


# In[4]:


driver_points = merged.groupby(['driverId', 'surname'])['points'].sum().reset_index()

top10 = driver_points.sort_values(by='points', ascending=False).head(10)


# In[5]:


# Plot bad
plt.figure(figsize=(10, 6))
sns.barplot(x='surname', y='points', data=top10)
plt.title('Top 10 F1 Drivers by Total Points (Bad Chart)')
plt.xlabel('Driver')
plt.ylabel('Total Points')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[6]:


driver_stats = merged.groupby(['driverId', 'surname']).agg({
    'points': 'sum',
    'raceId': 'count'
}).reset_index()


# In[7]:


driver_stats['points_per_race'] = driver_stats['points'] / driver_stats['raceId']

top10_ppr = driver_stats.sort_values('points_per_race', ascending=False).head(10)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(x='surname', y='points_per_race', data=top10_ppr, palette='crest')
plt.title('Top 10 Drivers by Points Per Race')
plt.ylabel('Avg. Points Per Race')
plt.xlabel('Driver')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[16]:


races = pd.read_csv('f1\\races.csv')
full = merged.merge(races[['raceId', 'year']], on='raceId')

hamilton = full[full['surname'] == 'Hamilton']
hamilton_season = hamilton.groupby('year')['points'].sum().reset_index()
plt.figure(figsize=(10,6))
plt.plot(hamilton_season['year'], hamilton_season['points'], marker='o')
plt.title('Lewis Hamilton: Total Points per Season')
plt.xlabel('Season')
plt.ylabel('Points')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[29]:


# Load required files and graph championship years as stars
driver_standings = pd.read_csv('f1\\driver_standings.csv')
drivers = pd.read_csv('f1\\drivers.csv')
races = pd.read_csv('f1\\races.csv')
results = pd.read_csv('f1\\results.csv')
constructors = pd.read_csv('f1\\constructors.csv')  

merged = driver_standings.merge(races[['raceId', 'year']], on='raceId')
merged = merged.merge(drivers[['driverId', 'surname']], on='driverId')

results_trim = results[['raceId', 'driverId', 'constructorId']].drop_duplicates()
merged = merged.merge(results_trim, on=['raceId', 'driverId'], how='left')

merged = merged.merge(constructors[['constructorId', 'name']], on='constructorId', how='left')
merged.rename(columns={'name': 'team'}, inplace=True)

top_3_drivers = ['Hamilton', 'Vettel', 'Verstappen']
start_years = {'Hamilton': 2007, 'Vettel': 2007, 'Verstappen': 2015} 
filtered = merged[merged['surname'].isin(top_3_drivers)]
filtered = filtered[filtered.apply(lambda row: row['year'] >= start_years[row['surname']], axis=1)]

points_by_season = (
    filtered.groupby(['surname', 'year', 'team'])['points']
    .sum()
    .reset_index()
)


teams = points_by_season['team'].dropna().unique()
team_colors = dict(zip(teams, sns.color_palette("husl", len(teams))))

plt.figure(figsize=(16, 8))
for driver in top_3_drivers:
    df_driver = points_by_season[points_by_season['surname'] == driver]
    for team in df_driver['team'].unique():
        df_team = df_driver[df_driver['team'] == team]
        label = f"{driver} ({team})"
        plt.plot(df_team['year'], df_team['points'], marker='o', label=label, color=team_colors[team])

#Add stars for championship years
championships = {
    'Hamilton': [2008, 2014, 2015, 2017, 2018, 2019, 2020],
    'Vettel': [2010, 2011, 2012, 2013],
    'Verstappen': [2021, 2022, 2023]
}

for driver, years in championships.items():
    for year in years:
        star = points_by_season[(points_by_season['surname'] == driver) & (points_by_season['year'] == year)]
        for _, row in star.iterrows():
            plt.plot(row['year'], row['points'], 'k*', markersize=12)

plt.title('Top 3 F1 Drivers: Points Per Season Colored by Team with Championship Years')
plt.xlabel('Season')
plt.ylabel('Total Points')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()


# In[20]:


driver_debuts = {
    'Hamilton': 2007,
    'Vettel': 2007,
    'Verstappen': 2015
} #To stop a data issue that causes visual issues

filtered_top3 = top3_seasonal[
    top3_seasonal.apply(lambda row: row['year'] >= driver_debuts[row['surname']], axis=1)
]

plt.figure(figsize=(12,6))

for driver in top3:
    driver_data = filtered_top3[filtered_top3['surname'] == driver]
    plt.plot(driver_data['year'], driver_data['points'], marker='o', label=driver)

plt.title('Top 3 F1 Drivers: Points Per Season')
plt.xlabel('Season')
plt.ylabel('Points')
plt.grid(True)
plt.legend(title='Driver')
plt.tight_layout()
plt.show()


# In[24]:


drivers = pd.read_csv('f1\\drivers.csv')

f1_points = pd.read_csv('f1\\driver_standings.csv')
merged = f1_points.merge(drivers[['driverId', 'surname']], on='driverId')

print(merged.columns)


# In[27]:


#Championship years
f1_points = pd.read_csv('f1\\driver_standings.csv')
drivers = pd.read_csv('f1\\drivers.csv')
races = pd.read_csv('f1\\races.csv')

merged = f1_points.merge(drivers[['driverId', 'surname']], on='driverId')
merged = merged.merge(races[['raceId', 'year']], on='raceId')

top_3_drivers = ['Hamilton', 'Vettel', 'Verstappen']
start_years = {'Hamilton': 2007, 'Vettel': 2007, 'Verstappen': 2015}
filtered_df = merged[merged['surname'].isin(top_3_drivers)]
filtered_df = filtered_df[filtered_df.apply(lambda row: row['year'] >= start_years[row['surname']], axis=1)]

season_points = filtered_df.groupby(['surname', 'year'])['points'].sum().reset_index()

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
for driver in top_3_drivers:
    driver_data = season_points[season_points['surname'] == driver]
    plt.plot(driver_data['year'], driver_data['points'], marker='o', label=driver)

# Championship years
championships = {
    'Hamilton': [2008, 2014, 2015, 2017, 2018, 2019, 2020],
    'Vettel': [2010, 2011, 2012, 2013],
    'Verstappen': [2021, 2022, 2023]
}

for driver, years in championships.items():
    champ_data = season_points[(season_points['surname'] == driver) & (season_points['year'].isin(years))]
    plt.plot(champ_data['year'], champ_data['points'], 'k*', markersize=12)

plt.title('Top 3 F1 Drivers: Total Points Per Season with Championship Years')
plt.xlabel('Season')
plt.ylabel('Total Points')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[30]:


#Regression line
from sklearn.linear_model import LinearRegression

driver_debuts = {
    'Hamilton': 2007,
    'Vettel': 2007,
    'Verstappen': 2015
}


filtered_top3 = top3_seasonal[
    top3_seasonal.apply(lambda row: row['year'] >= driver_debuts[row['surname']], axis=1)
]

plt.figure(figsize=(12, 6))

for driver in top3:
    driver_data = filtered_top3[filtered_top3['surname'] == driver]
    plt.plot(driver_data['year'], driver_data['points'], marker='o', label=driver)

    
    x = driver_data['year'].values.reshape(-1, 1)
    y = driver_data['points'].values.reshape(-1, 1)
    
    if len(x) > 1:  
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        plt.plot(driver_data['year'], y_pred, linestyle='--', alpha=0.5, label=f'{driver} Trend')

plt.title('Top 3 F1 Drivers: Points Per Season with Trend Lines')
plt.xlabel('Season')
plt.ylabel('Points')
plt.grid(True)
plt.legend(title='Driver')
plt.tight_layout()
plt.show()


# In[31]:


#Scatterplot
qualifying = pd.read_csv('f1\\qualifying.csv')

wins = results[results['positionOrder'] == 1].groupby('driverId').size().reset_index(name='wins')
poles = qualifying[qualifying['position'] == 1].groupby('driverId').size().reset_index(name='poles')

stats = drivers.merge(wins, on='driverId', how='left').merge(poles, on='driverId', how='left')
stats.fillna(0, inplace=True)

top_stats = stats.sort_values(by='wins', ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.scatterplot(data=top_stats, x='poles', y='wins', hue='surname', s=100)
plt.title('Pole Positions vs Race Wins (Top 15 Drivers)')
plt.xlabel('Pole Positions')
plt.ylabel('Race Wins')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:




