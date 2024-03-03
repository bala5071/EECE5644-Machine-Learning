import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from data_generator import postfix

# Setting up the dataset parameters
num_samples = 1000  # Total number of samples
noise_level = 0.01  # Variance of the noise
num_features = 40  # Number of features in each sample

# Generate the filename suffix based on dataset parameters
filename_suffix = postfix(num_samples, num_features, noise_level)

# Load the dataset
features = np.load(f"X{filename_suffix}.npy")
targets = np.load(f"y{filename_suffix}.npy")

print(f"Dataset contains n={features.shape[0]} samples, each with d={features.shape[1]} features, and {targets.shape[0]} corresponding labels.")

# Splitting the dataset into training and testing sets
features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.30, random_state=42)

print(f"Dataset split into {features_train.shape[0]} training and {features_test.shape[0]} testing samples.")

# Setting up the Lasso regression model
lasso_alpha = 0.1
lasso_regressor = Lasso(alpha=lasso_alpha)

# Configuring cross-validation
cv_strategy = KFold(n_splits=5, random_state=42, shuffle=True)

# Performing cross-validation to evaluate model performance
cv_scores = cross_val_score(
    lasso_regressor, features_train, targets_train, cv=cv_strategy, scoring="neg_root_mean_squared_error")

print(f"Cross-validation RMSE for α={lasso_alpha}: {-np.mean(cv_scores)} ± {np.std(cv_scores)}")

# Training the Lasso model on the entire training dataset
print("Training Lasso model on the entire training set...", end="")
lasso_regressor.fit(features_train, targets_train)
print(" completed")

# Calculating RMSE on training and testing sets
rmse_training = np.sqrt(rmse(targets_train, lasso_regressor.predict(features_train)))
rmse_testing = np.sqrt(rmse(targets_test, lasso_regressor.predict(features_test)))

print(f"Training RMSE = {rmse_training}, Testing RMSE = {rmse_testing}")