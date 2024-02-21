import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm

# Function to calculate Bhattacharyya coefficient
def bhattacharyya_coefficient(mean_0, covariance0, mean_1, covariance1):
    mean_differnce = mean_1 - mean_0
    Covariance_mean = (covariance0 + covariance1) / 2
    inv_Covariance_mean = np.linalg.inv(Covariance_mean)
    term_1 = np.dot(np.dot(mean_differnce.T, inv_Covariance_mean), mean_differnce) / 8
    term_2 = 0.5 * np.log(np.linalg.det(Covariance_mean) / np.sqrt(np.linalg.det(covariance0) * np.linalg.det(covariance1)))
    return term_1 + term_2

# Function to calculate Chernoff bound for a given beta
def chernoff_bound(beta, mean_0, covariance0, mean_1, covariance1):
    mean_differnce = mean_1 - mean_0
    term_1 = (1-beta) * beta * np.dot(np.dot(mean_differnce.T, np.linalg.inv((1-beta)*covariance0 + beta*covariance1)), mean_differnce)
    term_2 = 0.5 * np.log(np.linalg.det((1-beta)*covariance0 + beta*covariance1) / (np.linalg.det(covariance0)**(1-beta) * np.linalg.det(covariance1)**beta))
    return term_1 + term_2

mean_0 = np.array([-1, -1, -1, -1])
covariance0 = np.array([[5, 3, 1, -1], [3, 5, -2, -2], [1, -2, 6, 3], [-1, -2, 3, 4]])
mean_1 = np.array([1, 1, 1, 1])
covariance1 = np.array([[1.6, -0.5, -1.5, -1.2], [-0.5, 8, 6, -1.7], [-1.5, 6, 6, 0], [-1.2, -1.7, 0, 1.8]])

# To Calculate Bhattacharyya bound
Bh_coefficient = bhattacharyya_coefficient(mean_0, covariance0, mean_1, covariance1)
Bh_bound = np.exp(-Bh_coefficient)

# To Calculate Chernoff bound for beta in [0,1]
betas = np.linspace(0, 1, 100)
Chern_bounds = [chernoff_bound(beta, mean_0, covariance0, mean_1, covariance1) for beta in betas]
Chern_bounds_exp = np.exp(-np.array(Chern_bounds))

# Finding beta that minimizes Chernoff bound
min_Chern_bounds_index = np.argmin(Chern_bounds_exp)
optimal_beta = betas[min_Chern_bounds_index]
min_Chern_bounds = Chern_bounds_exp[min_Chern_bounds_index]

# Plotting Chernoff and Bhattacharyya bounds
plt.figure(figsize=(10, 6))
plt.plot(betas, Chern_bounds_exp, label='Chernoff Bound')
plt.axhline(y=Bh_bound, color='r', linestyle='--', label='Bhattacharyya Bound')
plt.scatter(optimal_beta, min_Chern_bounds, color='green', label=f'Optimal β = {optimal_beta:.2f}')
plt.xlabel('Beta (β)')
plt.ylabel('Bound on Error Probability')
plt.title('Chernoff and Bhattacharyya Bounds')
plt.legend()
plt.grid(True)
plt.show()

(optimal_beta, min_Chern_bounds, Bh_bound)

print(f"Optimal Beta: {optimal_beta:.4f}")
print(f"Minimum P(error) using Chernoff Bound: {min_Chern_bounds:.4f}")
print(f"Bhattacharyya Bound (Beta = 0.5): {Bh_bound:.4f}")
