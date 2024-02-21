import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to calculate the empirical error
def empirical_error(gamma, likelihood_ratios, lab):
    preds = likelihood_ratios > gamma
    return np.mean(preds != lab)

# Assuming you have loaded the samples and labels from Part A
samp = np.load('Samples.npy') 
lab = np.load('labels.npy') 

# True class conditional means
mean_vector0 = np.array([-1, -1, -1, -1])
mean_vector1 = np.array([1, 1, 1, 1])

# Separate the samp by class
samp_0 = samp[lab == 0]
samp_1 = samp[lab == 1]

# Calculate sample covariance matrices for each class
covariance_0_sample = np.cov(samp_0, rowvar=False)
covariance_1_sample = np.cov(samp_1, rowvar=False)

# Calculate the prior probabilities
priors0 = len(samp_0) / len(samp)
priors1 = len(samp_1) / len(samp)

# Estimate common covariance matrix using the sample average estimator
common_cov = priors0 * covariance_0_sample + priors1 * covariance_1_sample

# Function to calculate likelihood ratios with the common covariance matrix
def likelihood_ratio_common_cov(samp, mean_vector0, mean_vector1, common_cov):
    pdf_class0 = multivariate_normal.pdf(samp, mean=mean_vector0, cov=common_cov)
    pdf_class1 = multivariate_normal.pdf(samp, mean=mean_vector1, cov=common_cov)
    return pdf_class1 / pdf_class0

# Calculate likelihood ratios for all samp using the common covariance matrix
likelihood_ratios_common_cov = likelihood_ratio_common_cov(samp, mean_vector0, mean_vector1, common_cov)

# Optimize gamma to minimize the empirical error with the common covariance matrix
res_common_cov = minimize_scalar(empirical_error, bounds=(0, 1), args=(likelihood_ratios_common_cov, lab), method='bounded')
opti_gamma_common_cov = res_common_cov.x
minimum_error_common_cov = res_common_cov.fun

# Compute ROC curve with the common covariance matrix
fpr_common_cov, tpr_common_cov, _ = roc_curve(lab, likelihood_ratios_common_cov)
roc_auc_common_cov = auc(fpr_common_cov, tpr_common_cov)

# Plotting ROC curve for common covariance matrix
plt.figure(figsize=(10, 5))
plt.plot(fpr_common_cov, tpr_common_cov, label=f'Common Covariance ROC Curve (area = {roc_auc_common_cov:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Common Covariance Matrix')
plt.legend(loc="lower right")
plt.show()

print(f"Optimal Gamma (Common Covariance): {opti_gamma_common_cov:.4f}")
print(f"Minimum Empirical Error (Common Covariance): {minimum_error_common_cov:.4f}")
