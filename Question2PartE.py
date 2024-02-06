import numpy as np
import matplotlib.pyplot as plt

Priors_class0 = 0.35
Priors_class1 = 0.65

# To Define a range of B values
b_values = np.linspace(0, 10, 100)

# To Calculate the minimum-risk decision boundary gamma as a function of B
gamma_values = (Priors_class0 / Priors_class1) * b_values

# Plotting gamma as a function of B
plt.figure(figsize=(10, 5))
plt.plot(b_values, gamma_values, label='Minimum Risk Decision Boundary γ')
plt.xlabel('Cost B')
plt.ylabel('Decision Threshold γ')
plt.title('Minimum Risk Decision Boundary γ as a Function of B')
plt.legend()
plt.grid(True)
plt.show()

# Since we do not have the actual data distribution, we are not plotting the minimum expected risk.
# The calculation of the minimum expected risk would require knowledge of the distributions p(X|L=0) and p(X|L=1).
