import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
import sklearn.preprocessing as pp
import sklearn.metrics as skl_m

df = pd.read_csv('training_data_vt2025.csv')
#df.info()

# Modify the dataset, emphasizing different variables
df.iloc[:,12]=df.iloc[:,12]**2
df.iloc[:,13]=np.sqrt(df.iloc[:,13])
df.iloc[:,11] = df.iloc[:,11]**2

df['month_cos'] = np.cos(df.month*np.pi/12)
df['month_sin'] = np.sin(df.month*np.pi/12)

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

#df.iloc[:,15]=df.iloc[:,15].replace('low_bike_demand',False)
#df.iloc[:,15]=df.iloc[:,15].replace('high_bike_demand',True)
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

# Set up X,Y

# Train data 
X = train.iloc[:,0:-2]
Y = train['increase_stock']

# Test data
X_test = test.iloc[:,0:-2]
Y_test = test['increase_stock']

model = skl_lm.LogisticRegression()
# testar att skala datan

scaler = pp.StandardScaler().fit(X)
model.fit(scaler.transform(X),Y)
y_hat = model.predict(scaler.transform(X_test))

'''
# oskalad data
model.fit(X,Y)
y_hat = model.predict(X_test)'''

# Get confusion matrix
diff = pd.crosstab(y_hat, Y_test)
print(f'Confusion matrix: \n {diff}')

# Get recall, precision and accuracy:
# Given from formulas

TP = diff.iloc[0,0]
TN = diff.iloc[1,1]
FP = diff.iloc[1,0]
FN = diff.iloc[0,1]

#accuracy = (TP+TN)/(TP+TN+FP+FN)
accuracy = skl_m.accuracy_score(Y_test,y_hat,)
precision = skl_m.precision_score(Y_test,y_hat,pos_label='high_bike_demand')
recall = skl_m.recall_score(Y_test,y_hat,pos_label='high_bike_demand')
f1 = skl_m.f1_score(Y_test,y_hat,pos_label='high_bike_demand')
'''precision = TP/(TP+FP)
recall = TP/(TP+FN)
'''
print(f'''
RESULTS:
accuracy: {accuracy:.3f}
precision: {precision:.3f}
recall: {recall:.3f}
f1: {f1:.3f}''')