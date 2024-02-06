import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

# Set seed for reproducibility
np.random.seed(7)

# Improve readability for plots
plt.rc('font', size=22)
plt.rc('axes', titlesize=18, labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=10)
plt.rc('figure', titlesize=22)

# Load feature data and labels from the HAR dataset
fea_train_path = r"C:\Users\balas\Downloads\UCI HAR Dataset\train\X_train.txt"
lab_train_path = r'C:\Users\balas\Downloads\UCI HAR Dataset\train\y_train.txt'
fea_train = np.loadtxt(fea_train_path)
lab_train = np.loadtxt(lab_train_path)

fea_test_path = r'C:\Users\balas\Downloads\UCI HAR Dataset\test\X_test.txt'
lab_test_path = r'C:\Users\balas\Downloads\UCI HAR Dataset\test\y_test.txt'
fea_test = np.loadtxt(fea_test_path)
lab_test = np.loadtxt(lab_test_path)

# Normalizing the features to have zero mean and unit variance
scaler = StandardScaler()
fea_train_normalized = scaler.fit_transform(fea_train)
fea_test_normalized = scaler.transform(fea_test)  # Use the same scaler

# Apply PCA to reduce dimensionality for visualization and classification performance
pca = PCA(n_components=3)  # Reducing to 3 components for 3D visualization
x_har_train = pca.fit_transform(fea_train_normalized)
x_har_test = pca.transform(fea_test_normalized)

num_classes = len(np.unique(lab_train))
class_labels = np.unique(lab_train)

# To Compute mean vectors, covariance matrices, and class priors for Bayesian classifier
mean_vectors = []
covariance_matrices = []
regularization_param = 0.01  # Regularization parameter
class_priors = []

def classify_sample(sample, mean_vectors, covariance_matrices, class_priors, class_labels):
    posterior_probs = []
    for i, label in enumerate(class_labels):
        likelihood = multivariate_normal.pdf(sample, mean=mean_vectors[i], cov=covariance_matrices[i])
        posterior_prob = likelihood * class_priors[i]
        posterior_probs.append(posterior_prob)
    pred_label = class_labels[np.argmax(posterior_probs)]
    return pred_label


def calculate_loss(y_true, y_pred):
    return np.where(y_true == y_pred, 0, 1)

def calculate_average_expected_risk(y_true, y_pred):
    losses = calculate_loss(y_true, y_pred)
    return np.mean(losses)

# For Visualizing PCA projections for the training and test data in 3D
def visualize_3d_pca(features, labels, title):
    Fig = plt.figure(figsize=(10, 8))
    a = Fig.add_subplot(111, projection='3d')
    for label in class_labels:
        a.scatter(features[labels == label, 0], features[labels == label, 1], features[labels == label, 2],
                   label=f'Activity {label}')
    a.set_xlabel('PC1')
    a.set_ylabel('PC2')
    a.set_zlabel('PC3')
    a.legend()
    plt.title(title)
    plt.show()

for i in class_labels:
    class_data = x_har_train[lab_train == i]
    mean_vector = np.mean(class_data, axis=0)
    covariance_mat = np.cov(class_data.T) + regularization_param * np.eye(3)
    class_prior = class_data.shape[0] / x_har_train.shape[0]

    mean_vectors.append(mean_vector)
    covariance_matrices.append(covariance_mat)
    class_priors.append(class_prior)

# To Classify all samples in both training and test datasets
pred_lab_train = [classify_sample(sample, mean_vectors, covariance_matrices, class_priors, class_labels) for sample in x_har_train]
pred_lab_test = [classify_sample(sample, mean_vectors, covariance_matrices, class_priors, class_labels) for sample in x_har_test]

# To Convert lists to arrays for compatibility with sklearn's confusion_matrix
pred_lab_train = np.array(pred_lab_train)
pred_lab_test = np.array(pred_lab_test)

# To Calculate the Minimum Expected Risk for both datasets
average_expected_risk_train = calculate_average_expected_risk(lab_train, pred_lab_train)
average_expected_risk_test = calculate_average_expected_risk(lab_test, pred_lab_test)

print(f"Minimum Expected Risk for Training Data: {average_expected_risk_train:.4f}")
print(f"Minimum Expected Risk for Test Data: {average_expected_risk_test:.4f}")

# To Calculate confusion matrices
confusion_mat_train = confusion_matrix(lab_train, pred_lab_train)
confusion_mat_test = confusion_matrix(lab_test, pred_lab_test)
print("Confusion Matrix for Training Data:")
print(confusion_mat_train)
print("Confusion Matrix for Test Data:")
print(confusion_mat_test)

visualize_3d_pca(x_har_train, lab_train, "PCA Projections to 3D Space for Training Data")
visualize_3d_pca(x_har_test, lab_test, "PCA Projections to 3D Space for Test Data")