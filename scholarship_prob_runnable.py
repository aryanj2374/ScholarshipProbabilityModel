### FINAL VERSION ###

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd

# num_samples = 30000

# Generate synthetic data - need a scholarship database for better accuracy
# np.random.seed(42)
# GPA = np.random.uniform(1.0, 5.0, num_samples)  
# Income = np.random.uniform(10000, 500000, num_samples)  
# Extracurriculars = np.random.randint(0, 20, num_samples)  

# Eligibility criteria
# gpa_weight = 0.08
# income_weight = 0.2
# ec_weight = 0.015
# Eligible = ((gpa_weight * GPA) + (income_weight * (1 / Income)) + (ec_weight * Extracurriculars)) > 0.5 # noise: + (np.random.normal(0, 0.1, num_samples))

data = pd.read_csv('synthesized_scholarship_data.csv')

# One-hot encoding for categorical features
categorical_features = ['Demographic', 'Income']
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_features = one_hot_encoder.fit_transform(data[categorical_features].values)

# Combine numerical and encoded categorical features
numerical_features = data[['GPA', 'STEM', 'Athletics']].values
X = np.hstack([numerical_features, encoded_features])
y = data['Scholarship'].values

# Combine features into one array
# X = np.column_stack((GPA, Income, Extracurriculars))
# y = Eligible.astype(int)
# X = data[['GPA', 'STEM', 'Demographic', 'Income', 'Athletics']].values
# y = data['Scholarship'].values

# Normalizing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),  # Dropout to prevent overfitting
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model train
history = model.fit(X_train, y_train, epochs=50, batch_size=1875, validation_data=(X_test, y_test))

# Model evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plot training/validation acc graphs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training/validation loss graphs
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Predicts eligibility for user input
def predict_eligibility(gpa, stem, demographics, income, athletics):
    # Encode the categorical input features
    input_data_cat = one_hot_encoder.transform([[demographics, income]])
    input_data_num = np.array([[gpa, stem, athletics]])
    input_data = np.hstack([input_data_num, input_data_cat])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return f'{prediction[0][0]*100:.3f}% chance'

# Testing GPA, income, and extracurricular values
# gpa = 4.1
# income = 50000
# extracurriculars = 8
# print(predict_eligibility(gpa, income, extracurriculars))
gpa = 3.6
stem = 0
demographics = 'Other minority'
income = 'Low'
athletics = 0
print(predict_eligibility(gpa, stem, demographics, income, athletics))

# Taking user input
# exit = ""
while (exit != "Y"):
    gpa = float(input("Enter your GPA: "))
    stem = int(input("Are you in a STEM-related major? (0 or 1): ")) # 0, 1
    demographics = input("What demographic do you belong to: ") # White, Black, Asian, Hispanic, Other minority
    income = input("Which income class are you in: ") # Low, Middle, Upper
    athletics = int(input("Do you participate in a school sport (0 or 1): ")) # 0, 1
    print(predict_eligibility(gpa, stem, demographics, income, athletics))
    exit = input("Would you like to exit (Y/N): ")