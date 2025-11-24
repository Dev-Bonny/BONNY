Customer Churn Prediction using Artificial Neural Networks (ANN)
📘 Overview

This project builds an Artificial Neural Network (ANN) using Keras and TensorFlow to predict whether a bank customer will leave (churn) based on their personal and financial details.

The model learns from a dataset of 10,000 bank customers and predicts the likelihood of churn(customer leaving) using features such as credit score, age, account balance, and activity status.

🎯 Objective

To create a machine learning model that:

Analyzes bank customer data

Predicts the probability of churn (whether a customer will leave the bank)

Helps banks take proactive measures to retain valuable customers

📂 Dataset

File: Churn_Modelling.csv
Each row represents one customer.

Column	Description
CreditScore	Customer’s credit rating
Geography	Country of residence
Gender	Male / Female
Age	Age of the customer
Tenure	Number of years with the bank
Balance	Account balance
NumOfProducts	Number of bank products used
HasCrCard	Whether customer owns a credit card (1 = Yes, 0 = No)
IsActiveMember	Whether customer is active (1 = Yes, 0 = No)
EstimatedSalary	Customer’s annual salary
Exited	Target variable (1 = Left bank, 0 = Stayed)
⚙️ Installation

Clone the repository or download the script:

git clone https://github.com/Dev-Bonny/Churn-Prediction-ANN.git
cd Churn-Prediction-ANN


Install dependencies

pip install numpy pandas matplotlib scikit-learn keras tensorflow


(Optional) Ensure Python Scripts folder is added to PATH (e.g.):

C:\Users\Bonny\AppData\Roaming\Python\Python313\Scripts

🧩 Model Architecture
Layer	Details
Input Layer	11 neurons (for 11 features)
Hidden Layer 1	8 neurons, ReLU activation
Hidden Layer 2	8 neurons, ReLU activation
Dropout	20% to prevent overfitting
Output Layer	1 neuron, Sigmoid activation (binary output)

Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy
Epochs: 80
Batch Size: 16

🚀 Running the Program

Place the dataset file Churn_Modelling.csv in the same folder as the script.

Run the Python program:

python ai.py


The program will:

Preprocess the data

Train the neural network

Evaluate accuracy using a confusion matrix

the program churn.py

Predict whether a new customer will leave the bank

🔍 Example Prediction

The model predicts for a new customer:

Feature	Value
Geography	Los Angeles
Credit Score	800
Gender	Female
Age	40
Tenure	2
Balance	120,000
Number of Products	1
Has Credit Card	No
Is Active Member	Yes
Estimated Salary	70,000

Output Example:

Churn Probability: 0.18
Will the customer leave? No 😊

📊 Evaluation

The model’s performance is evaluated using:

Accuracy

Loss

Confusion Matrix

These metrics help determine how well the ANN predicts customer churn on unseen data.

🧰 Technologies Used

Python 3.13

NumPy

Pandas

Scikit-learn

Keras

TensorFlow

🧑‍💻 Author

Boniface (Dev-Bonny)
BSc. Computer Science Student | Aspiring Machine Learning Engineer
📧 [Your Email Here]
🌍 [GitHub Profile Link or Portfolio]
