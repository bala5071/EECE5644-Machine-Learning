import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# True class conditional means
mean_vector0 = np.array([-1, -1, -1, -1])
mean_vector1 = np.array([1, 1, 1, 1])

# True covariance matrices, adjusted to diagonal based on true variances
covariance0_diagonal = np.diag(np.diag([[5, 3, 1, -1], [3, 5, -2, -2], [1, -2, 6, 3], [-1, -2, 3, 4]]))
covariance1_diagonal = np.diag(np.diag([[1.6, -0.5, -1.5, -1.2], [-0.5, 8, 6, -1.7], [-1.5, 6, 6, 0], [-1.2, -1.7, 0, 1.8]]))

# Load the samples and labels, the same one as used in Part A)
samp = np.load('Samples.npy')
lab = np.load('labels.npy')

# Define likelihood ratio test function with diagonal covariance matrices
def likelihood_ratio_test_diagonal(samp, mean_vector0, covariance0, mean_vector1, covariance1, gamma):
    pdf_class0 = multivariate_normal.pdf(samp, mean=mean_vector0, cov=covariance0)
    pdf_class1 = multivariate_normal.pdf(samp, mean=mean_vector1, cov=covariance1)
    likelihood_ratio = pdf_class1 / pdf_class0
    preds = likelihood_ratio > gamma
    return preds, likelihood_ratio

def empirical_error(gamma, likelihood_ratios, lab):
    preds = likelihood_ratios > gamma
    return np.mean(preds!=lab)

# Compute likelihood ratios for all samp using diagonal covariance matrices
_, likelihood_ratios_diagonal = likelihood_ratio_test_diagonal(samp, mean_vector0, covariance0_diagonal, mean_vector1, covariance1_diagonal, gamma=1)

# Optimize gamma to minimize the empirical error with diagonal covariance matrices
res_diagonal = minimize_scalar(empirical_error, bounds=(0, 1), args=(likelihood_ratios_diagonal, lab), method='bounded')
opt_gamma_diagonal = res_diagonal.x
minimum_error_diagonal = res_diagonal.fun

# Compute ROC curve for diagonal covariance matrices
fpr_diagonal, tpr_diagonal, _ = roc_curve(lab, likelihood_ratios_diagonal)
roc_auc_diagonal = auc(fpr_diagonal, tpr_diagonal)

# Plotting ROC curve for diagonal covariance matrices
plt.figure(figsize=(10, 5))
plt.plot(fpr_diagonal, tpr_diagonal, label=f'Diagonal ROC Curve (area = {roc_auc_diagonal:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Diagonal Covariance Matrices')
plt.legend(loc="lower right")
plt.show()

print(f"Optimal Gamma (Diagonal): {opt_gamma_diagonal:.4f}")
print(f"Minimum Empirical Error (Diagonal): {minimum_error_diagonal:.4f}")