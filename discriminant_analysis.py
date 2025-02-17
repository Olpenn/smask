import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
import sklearn.preprocessing as pp
import sklearn.metrics as skl_m

df = pd.read_csv('training_data_vt2025.csv')
df.info()

df['month_cos'] = np.cos(df['month']*np.pi/12)
df['month_sin'] = np.sin(df['month']*np.pi/12)

# time of day, replaed with low,medium and high demand, 
# adding the new categories back in the end.
def categorize_demand(hour):
    if 20 <= hour or 7 >= hour:
        return 'night'
    elif 8 <= hour <= 14:
        return 'day'
    elif 15 <= hour <= 19:
        return 'evening'

df['demand_category'] = df['hour_of_day'].apply(categorize_demand)
df_dummies = pd.get_dummies(df['demand_category'], prefix='demand', drop_first=False)
df = pd.concat([df, df_dummies], axis=1)

# converting to bools
def if_zero(data):
    if data == 0:
        return True
    else:
        return False

# temperature

df['snowdepth_bool'] = df['snowdepth'].replace(0, False).astype(bool)
df['precip_bool'] = df['precip'].replace(0, False).astype(bool)

# Split into train and test:
np.random.seed(0)

df_modified=df[[#'holiday',
                'weekday',
                #'summertime',
                'temp',
                #'dew',
                #'humidity',
                'visibility',
                'windspeed',
                'month_cos',
                'month_sin',
                'demand_day',
                'demand_evening',
                'demand_night',
                'snowdepth_bool',
                'precip_bool',
                'increase_stock']]

N = df_modified.shape[0]
n = round(0.7*N)
trainI = np.random.choice(N,size=n,replace=False)
trainIndex = df_modified.index.isin(trainI)
train = df_modified.iloc[trainIndex]
test = df_modified.iloc[~trainIndex]


