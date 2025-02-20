import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv('training_data_vt2025.csv')

# modify the month to represent the periodicity that is observed in data.
df['month_cos'] = np.cos(df['month']*2*np.pi/12)
df['month_sin'] = np.sin(df['month']*2*np.pi/12)

# time of day, replaced with 3 bool values: is_night, is_day and is_evening, 
# adding the new categories back in the end.
def categorize_demand(hour):
    if 20 <= hour or 7 >= hour:
        return 'night'
    elif 8 <= hour <= 14:
        return 'day'
    elif 15 <= hour <= 19:
        return 'evening'

df['time_of_day'] = df['hour_of_day'].apply(categorize_demand)
df_dummies = pd.get_dummies(df['time_of_day'], prefix='is', drop_first=False)
df = pd.concat([df, df_dummies], axis=1)

# Create bool of snowdepth and percipitation
df['snowdepth_bool'] = df['snowdepth'].where(df['snowdepth'] == 0, 1)
df['precip_bool'] = df['precip'].where(df['precip'] == 0, 1)

# Seperate training data from target
X=df[[#'holiday',
        'weekday',
        #'summertime',
        'temp',
        #'dew',
        #'humidity',
        #'visibility',
        #'windspeed',
        #'month',
        'month_cos',
        'month_sin',
        #'hour_of_day',
        'is_day',
        'is_evening',
        'is_night',
        #'snowdepth_bool',
        'precip_bool'
        ]]

y=df['increase_stock']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Quadratic Discriminant Analysis (QDA)
qda = QuadraticDiscriminantAnalysis() 
X_train_lda = qda.fit(X_train, y_train)

# Make predictions
y_pred = qda.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))