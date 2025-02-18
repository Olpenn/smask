import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv('training_data_vt2025.csv')

# modify the month to represent the periodicity that is observed in data.
df['month_cos'] = np.cos(df['month']*np.pi/12)
df['month_sin'] = np.sin(df['month']*np.pi/12)

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
df['snowdepth_bool'] = df['snowdepth'].replace(0, False).astype(bool)
df['precip_bool'] = df['precip'].replace(0, False).astype(bool)

# Seperate training data from target
X=df[[#'holiday',
        'weekday',
        #'summertime',
        'temp',
        #'dew',
        #'humidity',
        #'visibility',
        #'windspeed',
        'month',
        #'month_cos',
        #'month_sin',
        #'hour_of_day',
        'is_day',
        'is_evening',
        'is_night',
        'snowdepth_bool',
        'precip_bool'
        ]]

y=df['increase_stock']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis(n_components=1) 
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Train a classifier (Logistic Regression)
clf = LogisticRegression()
clf.fit(X_train_lda, y_train)

# Make predictions
y_pred = clf.predict(X_test_lda)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))

"""
# Create a DataFrame showing correct and incorrect classifications
df2 = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})
df2 = pd.concat([X_test,df2], axis=1)

# Filter only misclassified rows
wrong_predictions = df2[df2['True Label'] != df2['Predicted Label']]
wrong_predictions = wrong_predictions.sort_values(by=['True Label', 'temp'])
print(wrong_predictions)

print(df.loc[958])
"""