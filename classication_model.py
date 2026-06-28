import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

# Generate Dataset
np.random.seed(42)
n = 1000
age = np.random.randint(18,60,n)
salary = np.random.randint(20000,120000,n)
time_spent = np.random.randint(1,30,n)
pages = np.random.randint(1,25,n)
previous = np.random.randint(0,10,n)
score = (
    salary/15000
    + time_spent*0.8
    + pages*1.2
    + previous*2
    - age*0.25
)

probability = 1/(1+np.exp(-(score-15)))
purchased = (probability>0.5).astype(int)
data = pd.DataFrame({
    "Age":age,
    "Salary":salary,
    "Time_Spent":time_spent,
    "Pages_Visited":pages,
    "Previous_Purchases":previous,
    "Purchased":purchased
})
print("\nFirst Five Records\n")
print(data.head())
print("\nDataset Shape:",data.shape)

# Features and Target
X = data.drop("Purchased",axis=1)
y = data["Purchased"]

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train,y_train)
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test,prediction)

print("\nAccuracy : {:.2f}%".format(accuracy*100))

# Classification Report
print("\nClassification Report\n")
print(classification_report(y_test,prediction))


# Confusion Matrix
cm = confusion_matrix(y_test,prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(model,X_test,y_test)
plt.title("ROC Curve")
plt.show()

# Feature Importance
importance = pd.Series(
    model.coef_[0],
    index=X.columns
)
importance.sort_values().plot(
    kind="barh",
    figsize=(8,5)
)
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.show()

# Sample Predictions
print("\nSample Predictions\n")
sample = pd.DataFrame({
    "Actual":y_test.values[:10],
    "Predicted":prediction[:10]
})
print(sample)