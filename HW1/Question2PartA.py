import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Setting the parameters for the Gaussian distributions and class priors
no_samp = 10000  
Prior_Prob0 = 0.35  
Prior_Prob1 = 0.65  
mean_vector0 = np.array([-1, -1, -1, -1])  
covariance_mat0 = np.array([[5, 3, 1, -1], [3, 5, -2, -2], [1, -2, 6, 3], [-1, -2, 3, 4]])  
mean_vector1 = np.array([1, 1, 1, 1])  
covariance_mat1 = np.array([[1.6, -0.5, -1.5, -1.2], [-0.5, 8, 6, -1.7], [-1.5, 6, 6, 0],
               [-1.2, -1.7, 0, 1.8]])  

# Initializing random seed for reproducibility
np.random.seed(0)


# Defining a function for likelihood ratio test
def likelihood_ratio_test(samp, mean_vector0, covariance_mat0, mean_vector1, covariance_mat1, gamma):
    pdf_class0 = multivariate_normal.pdf(samp, mean=mean_vector0, cov=covariance_mat0)
    pdf_class1 = multivariate_normal.pdf(samp, mean=mean_vector1, cov=covariance_mat1)
    likelihood_ratio = pdf_class1 / pdf_class0
    preds = likelihood_ratio > gamma
    return preds, likelihood_ratio


# Function to compute empirical error based on a given gamma
def empirical_error(gamma, likelihood_ratios, lab):
    preds = likelihood_ratios > gamma
    error = np.mean(preds != lab)
    return error

# Generating samples and their labels
samp, lab = [], []
for _ in range(no_samp):
    if np.random.rand() < Prior_Prob0: 
        samp.append(multivariate_normal.rvs(mean=mean_vector0, cov=covariance_mat0))
        lab.append(0)
    else:  
        samp.append(multivariate_normal.rvs(mean=mean_vector1, cov=covariance_mat1))
        lab.append(1)
samp = np.array(samp)
lab = np.array(lab)
np.save('Samples',samp)
np.save('Labels',lab)
print("Samples shape:", samp.shape)
print("Labels shape:", lab.shape)

# Computing likelihood ratios
_, likelihood_ratios = likelihood_ratio_test(samp, mean_vector0, covariance_mat0, mean_vector1, covariance_mat1, gamma=1)

# Finding the optimal gamma by minimizing the empirical error
results = minimize_scalar(empirical_error, bounds=(0, 1), args=(likelihood_ratios, lab), method='bounded')
optimal_gamma = results.x
min_error = results.fun
print("Optimal Gamma:", optimal_gamma)
print("Minimum Error Rate:", min_error)

# To Plot ROC Curve
fpr, tpr, thresholds = roc_curve(lab, likelihood_ratios)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(fpr[thresholds > optimal_gamma][-1], tpr[thresholds > optimal_gamma][-1], color='red',
            label=f'Optimal γ = {optimal_gamma:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Finding Error Rate vs Gamma Plot
gammas = np.linspace(0, 1, 500)
errors = [empirical_error(gamma, likelihood_ratios, lab) for gamma in gammas]
plt.figure(figsize=(10, 5))
plt.plot(gammas, errors, label='Empirical Error Rate')
plt.scatter(optimal_gamma, min_error, color='red', label=f'Minimum Error @ γ = {optimal_gamma:.3f}')
plt.xlabel('Gamma')
plt.ylabel('Error Rate')
plt.title('Error')
plt.show()
