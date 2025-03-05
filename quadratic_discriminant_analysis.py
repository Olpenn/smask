from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from preprocessing import get_data

X, Y, X_test, Y_test = get_data()

# Has a variance of 0
X = X.drop(columns='snowdepth_bool')
X_test = X_test.drop(columns='snowdepth_bool')

# Apply Quadratic Discriminant Analysis (QDA)
qda = QuadraticDiscriminantAnalysis() 
X_train_lda = qda.fit(X, Y)

# Make predictions
y_pred = qda.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print(classification_report(Y_test, y_pred))