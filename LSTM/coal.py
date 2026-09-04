# Import NumPy for numerical computations and array manipulation
import numpy as np

# Import Pandas for reading and manipulating datasets
import pandas as pd

# Import TensorFlow for building and training neural networks
import tensorflow as tf

# Import MinMaxScaler to normalize values between 0 and 1
from sklearn.preprocessing import MinMaxScaler

# Import evaluation metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score


# -------------------------------------------------------
# LOAD AND PREPROCESS DATA
# -------------------------------------------------------

# Read CSV file containing monthly coal consumption data
data = pd.read_csv('MonthlyCoalConsumption.csv')

# Extract the Value column into a NumPy array
values = np.array(data['Value'])

# Normalize values to range (0–1)
# This improves training speed and model stability
scaler = MinMaxScaler(feature_range=(0, 1))

# Reshape values into column form then normalize
scaled_values = scaler.fit_transform(values.reshape(-1, 1))


# -------------------------------------------------------
# CREATE INPUT SEQUENCES
# -------------------------------------------------------

# Define the number of previous months used for prediction
window_size = 12

# Lists for storing training sequences
X = []
y = []

# Generate sequential windows
for i in range(len(scaled_values) - window_size):

    # Store previous 12 months
    X.append(scaled_values[i:i+window_size])

    # Store target month
    y.append(scaled_values[i+window_size])

# Convert lists into arrays
X = np.array(X)
y = np.array(y)


# -------------------------------------------------------
# SPLIT DATA INTO TRAINING AND TEST SETS
# -------------------------------------------------------

# Use 80% data for training
split = int(0.8 * len(X))

# Training dataset
X_train = X[:split]

# Testing dataset
X_test = X[split:]

# Training labels
y_train = y[:split]

# Testing labels
y_test = y[split:]


# -------------------------------------------------------
# DEFINE MODEL ARCHITECTURE
# -------------------------------------------------------

# JUSTIFICATION:
# LSTM is selected instead of SimpleRNN because:
# 1. It remembers long-term patterns.
# 2. Handles seasonal time-series data effectively.
# 3. Prevents vanishing gradient issues.

model = tf.keras.Sequential([

    # LSTM layer with 32 memory cells
    tf.keras.layers.LSTM(
        units=32,
        activation='tanh',
        input_shape=(window_size, 1)
    ),

    # Output layer for prediction
    tf.keras.layers.Dense(1)
])


# -------------------------------------------------------
# COMPILE MODEL
# -------------------------------------------------------

# Configure model training:
# Adam → adaptive optimizer
# MSE → regression loss
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)


# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------

# Train for 50 iterations
# Batch size = 16 samples at a time
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16
)


# -------------------------------------------------------
# EVALUATE MODEL LOSS
# -------------------------------------------------------

# Calculate loss on unseen test data
loss = model.evaluate(X_test, y_test)

print("Test Loss:", loss)


# -------------------------------------------------------
# GENERATE PREDICTIONS
# -------------------------------------------------------

predictions = model.predict(X_test)


# -------------------------------------------------------
# CONVERT REGRESSION OUTPUT TO BINARY
# -------------------------------------------------------

# Threshold at 0.5 for metric calculation
y_pred_binary = (predictions > 0.5).astype(int)

y_test_binary = (y_test > 0.5).astype(int)


# -------------------------------------------------------
# CALCULATE PRECISION
# -------------------------------------------------------

# Precision:
# Measures proportion of predicted positives
# that are actually correct

precision = precision_score(
    y_test_binary,
    y_pred_binary
)

print("Precision:", precision)


# -------------------------------------------------------
# CALCULATE RECALL
# -------------------------------------------------------

# Recall:
# Measures how many actual positives
# are correctly identified

recall = recall_score(
    y_test_binary,
    y_pred_binary
)

print("Recall:", recall)


# -------------------------------------------------------
# CALCULATE AUC
# -------------------------------------------------------

# AUC:
# Measures model’s ability to distinguish classes

auc = roc_auc_score(
    y_test_binary,
    predictions
)

print("AUC:", auc)