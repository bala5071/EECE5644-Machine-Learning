# Importing required libraries
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from data_generator import postfix, liftDataset
import matplotlib.pyplot as plt

# Defining dataset characteristics
total_samples = 1000  # Number of samples in the dataset
features_dim = 40  # Number of features per sample
noise_std_dev = 0.01  # Standard deviation of noise added to the dataset

# Generating a unique identifier for dataset files based on their characteristics
dataset_suffix = postfix(total_samples, features_dim, noise_std_dev)

# Loading the dataset files based on the generated suffix
unlift_X = np.load(f"X{dataset_suffix}.npy")
targets = np.load(f"y{dataset_suffix}.npy")

# Displaying information about the loaded dataset
print(f"Dataset loaded with {unlift_X.shape[0]} samples, each containing {unlift_X.shape[1]} features. Total targets: {targets.shape[0]}.")

# Splitting the dataset into training and test sets
features_train, features_test, targets_train, targets_test = train_test_split(unlift_X, targets, test_size=0.30, random_state=42)
print(f"Dataset divided into {features_train.shape[0]} training samples and {features_test.shape[0]} test samples.")

# Initializing the Linear Regression model
linear_regressor = LinearRegression()

# Fitting the model on the training data
print("Training the Linear Regression model...", end="")
linear_regressor.fit(features_train, targets_train)
print(" completed.")

# Evaluating the model's performance with RMSE on both training and test data
rmse_training = np.sqrt(mean_squared_error(targets_train, linear_regressor.predict(features_train)))
rmse_testing = np.sqrt(mean_squared_error(targets_test, linear_regressor.predict(features_test)))
print(f"Model evaluation metrics: Training RMSE = {rmse_training:.5f}, Test RMSE = {rmse_testing:.5f}")

# Displaying the model's parameters
print("Linear Regression model coefficients:")
print(f"Intercept: {linear_regressor.intercept_:.5f}", end="")
for index, coefficient in enumerate(linear_regressor.coef_):
    print(f", Coefficient {index}: {coefficient:.5f}", end="")
print()

