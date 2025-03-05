import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.preprocessing as pp
import sklearn.metrics as skl_m

import sklearn.neighbors as skl_nb


# ------------- DATA MODIFICATION AND SETUP --------------

df = pd.read_csv('training_data_vt2025.csv')

# Modify the dataset, emphasizing different variables

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
np.random.seed(0)

df_modified=df[[#'holiday',
                'weekday',
                #'summertime',
                'temp',
                #'dew',
                'humidity',
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


# --------------------------- K-NN METHOD ---------------------------

"""
# Tests for k-value
# TEST 1 - uniform distance
missclassification = []
for k in range(500): # Try n_neighbours = 1, 2, ....,

    #kNN method
    scaler = pp.StandardScaler().fit(X)
    model = skl_nb.KNeighborsClassifier(n_neighbors = k+1, weights = 'uniform')
    model.fit(scaler.transform(X),Y)

    # Prediction
    y_hat = model.predict(scaler.transform(X_test))
    missclassification.append(np.mean(y_hat != Y_test))

K = np.linspace(1, 500, 500)
plt.plot(K, missclassification, '.')
plt.ylabel('Missclassification')
plt.xlabel('Number of neighbours')
plt.show()

#TEST 2
missclassification = []
for k in range(500): # Try n_neighbours = 1, 2, ....,

    #kNN method
    scaler = pp.StandardScaler().fit(X)
    model = skl_nb.KNeighborsClassifier(n_neighbors = k+1, weights = 'distance')
    model.fit(scaler.transform(X),Y)

    # Prediction
    y_hat = model.predict(scaler.transform(X_test))
    missclassification.append(np.mean(y_hat != Y_test))

K = np.linspace(1, 500, 500)
plt.plot(K, missclassification, '.')
plt.ylabel('Missclassification')
plt.xlabel('Number of neighbours')
plt.show()
"""

# creating the model
model = skl_nb.KNeighborsClassifier(n_neighbors = 120, weights = 'distance')

# Scaling the data, otherwise
scaler = pp.StandardScaler().fit(X)
model.fit(scaler.transform(X),Y)
y_hat = model.predict(scaler.transform(X_test))

# Get confusion matrix
diff = pd.crosstab(y_hat, Y_test)
print(f'Confusion matrix for K-NN: \n {diff}')

# Get metrics:
print(f'Metrics for K-NN: \n{skl_m.classification_report(Y_test, y_hat)}')



# --------------- LDA & QDA ------------------

# LDA:

# Apply Linear Discriminant Analysis (LDA)
lda = skl_da.LinearDiscriminantAnalysis(n_components=1) 
X_train_lda = lda.fit_transform(X, Y)
X_test_lda = lda.transform(X_test)

# Train a classifier (Logistic Regression)
clf = skl_lm.LogisticRegression()
clf.fit(X_train_lda, Y)

# Make predictions
y_pred = clf.predict(X_test_lda)

# Evaluate accuracy
accuracy = skl_m.accuracy_score(Y_test, y_pred)
print(f"Model Accuracy for LDA: \n {accuracy:.2f}")

# Get confusion matrix
diff = pd.crosstab(y_pred, Y_test)
print(f'Confusion matrix for LDA: \n {diff}')

# QDA: 

# Has a variance of 0
X = X.drop(columns='snowdepth_bool')
X_test = X_test.drop(columns='snowdepth_bool')

# Apply Quadratic Discriminant Analysis (QDA)
qda = skl_da.QuadraticDiscriminantAnalysis() 
X_train_lda = qda.fit(X, Y)

# Make predictions
y_pred = qda.predict(X_test)

# Evaluate accuracy
accuracy = skl_m.accuracy_score(Y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print(f'Metrics for QDA: \n{skl_m.classification_report(Y_test, y_pred)}')

# --------------- LOGISTIC REGRESSION ---------------

# Scaling the data, otherwise
scaler = pp.StandardScaler().fit(X)
model.fit(scaler.transform(X),Y)
y_hat = model.predict(scaler.transform(X_test))

# Get confusion matrix
diff = pd.crosstab(y_hat, Y_test)
print(f'Confusion matrix: \n {diff}')

# Get metrics:
print(f'metrics for Logistic regression: \n {skl_m.classification_report(Y_test, y_hat)}')


# ---------------- TREE-BASED METHODS ---------------

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [10, 20, None],     
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 4]     
}

# Set up Grid Search
random_search = RandomizedSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit on training data
random_search.fit(X, Y)

# Get the best hyperparameters
print("Best Parameters: ", random_search.best_params_)
print("Best Accuracy: %.2f" % random_search.best_score_)

# Update the model with the best parameters
best_model = random_search.best_estimator_

# Fit the best model on the training data
best_model.fit(X, Y)

# Make predictions using the optimized model
y_predict = best_model.predict(X_test)

print(f'Metrics for tree method: \n{skl_m.classification_report(Y_test, y_predict)}')

# ------------- NAIVE MODEL -----------------

# We don't train any model, we just guess there is always a low demand
y_test = test['increase_stock']
y_predict = np.array(["low_bike_demand"]*len(y_test))

print(f'Metrics for naive model: \n{skl_m.classification_report(y_test, y_predict)}')