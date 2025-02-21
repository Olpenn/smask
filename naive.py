import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Import data set
df = pd.read_csv('training_data_vt2025.csv')

# Split into "train" and "test" as in the other models
np.random.seed(0)
N = df.shape[0]
n = round(0.7*N)
trainI = np.random.choice(N,size=n,replace=False)
trainIndex = df.index.isin(trainI)
train = df.iloc[trainIndex]
test = df.iloc[~trainIndex]

# We don't train any model, we just guess there is always a low demand
y_test = test['increase_stock']
y_predict = np.array(["low_bike_demand"]*len(y_test))

print(classification_report(y_test, y_predict))
