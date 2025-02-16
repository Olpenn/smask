import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm

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
    if 20 <= hour <= 7:
        return 'low_demand'
    elif 8 <= hour <= 14:
        return 'medium_demand'
    elif 15 <= hour <= 19:
        return 'high_demand'

df['demand_category'] = df['hour'].apply(categorize_demand)
df_dummies = pd.get_dummies(df['demand_category'], prefix='demand', drop_first=False)
df = pd.concat([df, df_dummies], axis=1)



# Split into train and test:

#df.iloc[:,15]=df.iloc[:,15].replace('low_bike_demand',False)
#df.iloc[:,15]=df.iloc[:,15].replace('high_bike_demand',True)
np.random.seed(0)

N = df.shape[0]
n = round(0.5*N)
trainI = np.random.choice(N,size=n,replace=False)
trainIndex = df.index.isin(trainI)
train = df.iloc[trainIndex]
test = df.iloc[~trainIndex]

# Set up X,Y

# Train data 
X = train.iloc[:,0:14]
Y = train.iloc[:,15]

# Test data
X_test = test.iloc[:,0:14]
Y_test = test.iloc[:,15]

model = skl_lm.LogisticRegression()
model.fit(X,Y)
y_hat = model.predict(X_test)
# needs modification, very bad and not converging.

# Get confusion matrix
diff = pd.crosstab(y_hat, Y_test)
print(f'Confusion matrix: \n {diff}')

# Get recall, precision and accuracy:
# Given from formulas

TP = diff.iloc[0,0]
TN = diff.iloc[1,1]
FP = diff.iloc[1,0]
FN = diff.iloc[0,1]

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print(f'''accuracy: {accuracy}
    precision: {precision}
    recall: {recall}''')