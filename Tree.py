import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report

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

X_train = train.drop(columns=['increase_stock'])
# Need to transform the qualitative variables to dummy variables 

y_train = train['increase_stock']

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
random_search.fit(X_train, y_train)

# Get the best hyperparameters
print("Best Parameters: ", random_search.best_params_)
print("Best Accuracy: %.2f" % random_search.best_score_)

# Update the model with the best parameters
best_model = random_search.best_estimator_

# Fit the best model on the training data
best_model.fit(X_train, y_train)

# Make predictions using the optimized model




###
#dot_data = tree.export_graphviz(model, out_file=None, feature_names = X_train.columns,class_names = model.classes_, 
#                                filled=True, rounded=True,leaves_parallel=True, proportion=True)
#graph = graphviz.Source(dot_data)
#graph.render("decision_tree", format="pdf")
X_test = test.drop(columns=['increase_stock'])
y_test = test['increase_stock']
y_predict = best_model.predict(X_test)



print(classification_report(y_test, y_predict))
