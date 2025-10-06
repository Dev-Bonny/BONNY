# ---------------------------
# Importing the libraries
# ---------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout

# ---------------------------
# Importing the dataset
# ---------------------------
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# ---------------------------
# Encoding categorical data
# ---------------------------
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# One-hot encode Geography
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype=np.float64)

# Avoid dummy variable trap
X = X[:, 1:]

# ---------------------------
# Splitting the dataset
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---------------------------
# Feature Scaling
# ---------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------------------------
# Building the ANN
# ---------------------------
classifier = Sequential()

# Input layer + first hidden layer
classifier.add(Dense(units=8, activation='relu', input_dim=11))
classifier.add(Dropout(rate=0.2))  # helps prevent overfitting

# Second hidden layer
classifier.add(Dense(units=8, activation='relu'))
classifier.add(Dropout(rate=0.2))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# ---------------------------
# Compiling the ANN
# ---------------------------
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------------
# Training the ANN
# ---------------------------
classifier.fit(X_train, y_train, batch_size=16, epochs=80)

# ---------------------------
# Predicting the Test set
# ---------------------------
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ---------------------------
# Predicting a new customer
# ---------------------------
"""Predict if the customer with the following information will leave:
Geography: Los Angeles
Credit Score: 800
Gender: Female
Age: 40
Tenure: 2
Balance: 120000
Number of Products: 1
Has Credit Card: No
Is Active Member: Yes
Estimated Salary: 70000
"""

# For Geography (Los Angeles is a new category — treat as "Other")
# Assume it maps to [0, 0] since one-hot encoding keeps known geographies

new_data = np.array([[0.0, 0, 800, 0, 40, 2, 120000, 1, 0, 1, 70000]])
new_data = sc.transform(new_data)
new_prediction = classifier.predict(new_data)
result = (new_prediction > 0.5)

print(f"\nChurn Probability: {new_prediction[0][0]:.2f}")
print("Will the customer leave?", "Yes 😞" if result else "No 😊")
