import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from data_generator import postfix, lift, liftDataset



# Number of samples
N = 1000

# Noise variance 
sigma = 0.01


# Feature dimension
d = 40


psfx = postfix(N,d,sigma) 
      

unlift_X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")

print("Dataset has n=%d samples, each with d=%d features," % unlift_X.shape,"as well as %d labels." % y.shape[0])

X = np.array(liftDataset(unlift_X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))

fractions = np.arange(0.1, 1.1, 0.1)
train_rmse = []
test_rmse = []

for f in fractions:
    fraction_Size = int(X_train.shape[0]*f)
    X_train_frac = X_train[0:fraction_Size,:]
    y_train_frac = y_train[0:fraction_Size]
    mod = LinearRegression()
    mod.fit(X_train_frac, y_train_frac)
    y_train_predict = mod.predict(X_train)
    y_test_predict = mod.predict(X_test)
    train_rmse.append(np.sqrt(rmse(y_train, y_train_predict)))
    test_rmse.append(np.sqrt(rmse(y_test, y_test_predict)))

    # Compute RMSE on train and test sets
    rmse_train = rmse(y_train,mod.predict(X_train))
    rmse_test = rmse(y_test,mod.predict(X_test))
    print(f"Training with {f*100:.0f}% of the data.")
    print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))


       

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(fractions, train_rmse, label='Train RMSE')
plt.plot(fractions, test_rmse, label='Test RMSE')
plt.xlabel('Fraction of Training Set')
plt.ylabel('RMSE')
plt.title('RMSE vs. Training Set Size')
plt.legend()
plt.show()
