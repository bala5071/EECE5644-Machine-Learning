import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy import linalg as LA

# Load the Dataset from local storage
df = pd.read_csv(r'C:\Users\balas\Downloads\wine+quality\winequality-white.csv', delimiter=';')
data = df.to_numpy()
n_samples = data.shape[0]  # No of Samples
class_labels = df.iloc[:, -1].values  # Access the quality column for labels
features = df.iloc[:, :-1].values
n_classes = 11  # No of classes
n_features = 11  # No of features
mean_vectors = np.zeros(shape=[n_classes, n_features])
Covariance_mat = np.zeros(shape=[n_classes, n_features, n_features])

# Compute Mean Vectors and Covariance Matrices
for i in range(n_classes):
    class_data = features[(class_labels == i)]
    mean_vectors[i, :] = np.mean(class_data, axis=0)

    
    if i not in class_labels:
        Covariance_mat[i, :, :] = np.eye(n_features)
    else:
        Covariance_mat[i, :, :] = np.cov(class_data, rowvar=False)
        regularization_term = (0.000000005) * (
                    np.trace(Covariance_mat[i, :, :]) / LA.matrix_rank(Covariance_mat[i, :, :])) * np.eye(
            n_features)
        Covariance_mat[i, :, :] += regularization_term

# Loss Matrix for 0-1 Loss
loss_matrix = np.ones(shape=[n_classes, n_classes]) - np.eye(n_classes)

# Compute Class-Conditional PDF
class_conditional_pdfs = np.zeros(shape=[n_classes, n_samples])
for i in range(n_classes):
    if i in class_labels:
        class_conditional_pdfs[i, :] = multivariate_normal.pdf(features, mean=mean_vectors[i, :],
                                                               cov=Covariance_mat[i, :, :])

# Estimating the Class Priors
class_priors = np.zeros(shape=[n_classes, 1])
for i in range(n_classes):
    class_priors[i] = np.size(class_labels[np.where((class_labels == i))]) / n_samples

# Assuming class_priors is shaped as (11, 1), we directly use class_priors without transposing
# Ensure total_probability is a 1D array by summing class_conditional_pdfs multiplied by class_priors across classes
tol_probability = np.sum(class_conditional_pdfs * class_priors, axis=0)

# Class_posteriors are computed based on Class Priors
class_posteriors = (class_conditional_pdfs * class_priors) / tol_probability

# Evaluate Expected Risk for Minimum Risk Decision Making
exp_risk = np.matmul(loss_matrix, class_posteriors)
decisions = np.argmin(exp_risk, axis=0)
print("Minimum Expected Risk:", np.sum(np.min(exp_risk, axis=0)) / n_samples)

# Estimating Confusion Matrix
confusion_mat = np.zeros(shape=[n_classes, n_classes])
for decision in range(n_classes):
    for actual_label in range(n_classes):
        if actual_label in class_labels and decision in class_labels:
            confusion_mat[decision, actual_label] = np.size(
                np.where((decision == decisions) & (actual_label == class_labels))) / np.size(
                np.where(class_labels == actual_label))

print("Confusion Matrix:")
print(confusion_mat)

# To Plot the Data Distribution
Figure_plot = plt.figure()
ax = plt.axes(projection="3d")
for i in range(n_classes):
    ax.scatter(features[(class_labels == i), 1], features[(class_labels == i), 2], features[(class_labels == i), 3],
               label=f'Class {i}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.title('White Wine Quality Data Distribution')
plt.show()
