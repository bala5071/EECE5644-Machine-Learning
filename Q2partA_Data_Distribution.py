import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(threshold=np.inf)
plt.rcParams['figure.figsize'] = [7,7] #Plot size

N_f = 4          #Features  
N_S = 10000       #Samples
N_l = 2            #Labels  

# Mean vectors
mean = np.ones(shape=[N_l, N_f])
mean [0, :] = [-1,-1,-1,-1]                      

# Covariance matrices
covariance = np.ones(shape=[N_l, N_f, N_f])            
covariance [0, :, :] = [[2, -0.5, 0.3, 0],[-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]]
covariance [1, :, :] = [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]
np.random.seed(10)
priors = [0.35, 0.65] 
label = (np.random.rand(N_S) >= priors[1]).astype(int)

# Generate Gaussian distribution for 10000 samples using mean and covariance matrices
X = np.zeros(shape = [N_S, N_f])
for j in range(N_S): 
        if (label[j] == 0):
                X[j, :] = np.random.multivariate_normal(mean[0, :], covariance[0, :,:])
        elif (label[j] == 1):
                X[j, :] = np.random.multivariate_normal(mean[1, :], covariance[1, :,:])

GPDF0 = np.log(multivariate_normal.pdf(X,mean = mean[0, :], cov = covariance[0, :,:]))  #using the matrices for creation of GPDF
GPDF1 = np.log(multivariate_normal.pdf(X,mean = mean[1, :], cov = covariance[1, :,:]))
discrim_score = GPDF1 - GPDF0
sorted_tau = np.sort(discrim_score)
tau_sweep = []

for i in range(0,9999):
        tau_sweep.append((sorted_tau[i] + sorted_tau[i+1])/2.0)
        
decision = []
TP = [None] * len(tau_sweep)
FP = [None] * len(tau_sweep)
minPerror = [None] * len(tau_sweep)

for (index, tau) in enumerate(tau_sweep):
        decision = (discrim_score >= tau)
        TP[index] = (np.size(np.where((decision == 1) & (label == 1))))/np.size(np.where(label == 1))
        FP[index] = (np.size(np.where((decision == 1) & (label == 0))))/np.size(np.where(label == 0))
        minPerror[index] = (priors[0] * FP[index]) + (priors[1] * (1 - TP[index]))

gamma_ideal = np.log(priors[0] / priors[1]) #Classify using class priors
ideal_decision = (discrim_score >= gamma_ideal)
TP_ideal = (np.size(np.where((ideal_decision == 1) & (label == 1))))/np.size(np.where(label == 1))
FP_ideal = (np.size(np.where((ideal_decision == 1) & (label == 0))))/np.size(np.where(label == 0))
minPerror_ideal = (priors[0] * FP_ideal) + (priors[1] * (1 - TP_ideal))
print("Gamma Ideal - %f and corresponding minimum error %f" %(np.exp(gamma_ideal), minPerror_ideal))

#Plot The Data Distribution
fig = plt.figure()
ax = plt.axes(projection = "3d")
Cls0 = ax.scatter(X[(label==0),3],X[(label==0),1],X[(label==0),2],'+',color ='orange', label="0")
Cls1 = ax.scatter(X[(label==1),3],X[label==1,1],X[label==1,2],'.',color = 'blue', label="1")
plt.xlabel('X3')
plt.ylabel('X1')
ax.set_zlabel('X2')
ax.legend()
plt.title('The Data Distribution')
plt.show()