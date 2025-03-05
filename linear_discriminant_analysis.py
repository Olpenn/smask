import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from preprocessing import get_data

X, Y, X_test, Y_test = get_data()

# Apply Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis(n_components=1) 
X_train_lda = lda.fit_transform(X, Y)
X_test_lda = lda.transform(X_test)

# Train a classifier (Logistic Regression)
clf = LogisticRegression()
clf.fit(X_train_lda, Y)

# Make predictions
y_pred = clf.predict(X_test_lda)

# Evaluate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print(classification_report(Y_test, y_pred))
