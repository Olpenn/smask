import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


data = pd.read_csv('/Users/jakobhanson/smask/training_data_vt2025.csv')
data.info()

# sampling indices for training
trainIndex = np.random.choice(data.shape[0], size=1200, replace=False)
train = data.iloc[trainIndex] # training set 
test = data.iloc[~data.index.isin(trainIndex)] #Test set 

X_train = train.drop(columns=['increase_stock'])
# Need to transform the qualitative variables to dummy variables 

y_train = train['increase_stock']

model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200, 500, 10000],  # Number of trees
    'max_depth': [5, 10, 20, 30, None],       # Tree depth (None means unlimited)
    'min_samples_split': [1,2,3,4,5,6],          # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 3, 4, 5],        # Minimum samples required at each leaf node
}

# Set up Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit on training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: %.2f" % grid_search.best_score_)

model.fit(X=X_train, y=y_train)


###
#dot_data = tree.export_graphviz(model, out_file=None, feature_names = X_train.columns,class_names = model.classes_, 
#                                filled=True, rounded=True,leaves_parallel=True, proportion=True)
#graph = graphviz.Source(dot_data)
#graph.render("decision_tree", format="pdf")
X_test = test.drop(columns=['increase_stock'])
y_test = test['increase_stock']
y_predict = model.predict(X_test)



print(classification_report(y_test, y_predict))
