#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt


#construct a table with the data
path = os.getcwd() + '/data/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
#data.head()
#print (data)

#plot for training data
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
#plt.show()


# sigmoid function
def sigmoid (z):
    return 1 / (1 + np.exp(-z))

# cost function to evaulate the solution
def cost_function(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    #classification classes
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# finding the gradient of a single data point
def gradient (theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int (theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad

# implements the classifier 
def predict (theta, X):
    probability = sigmoid (X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


## perfom calculations and find the gradient descent

# to make matrix multiplication easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

#checking the dimension of the matrices
#print (X.shape, theta.shape, y.shape)

# calculate the gradient descent
result = opt.fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(X, y))
optimal_value = result[0]

# testing 
theta_min = np.matrix(optimal_value)
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) / len (correct)) * 100
print ('accuracy = {0}%'.format(accuracy))



#Plotting the decision boundary
theta_arr = np.squeeze(np.asarray(theta_min))
plot_x = np.array([min(data.iloc[:, 1]) - 2, max(data.iloc[:, 2]) + 2])
plot_y = (- 1.0 / theta_arr[2]) * (theta_arr[1] * plot_x + theta_arr[0])
ax.plot(plot_x, plot_y)
ax.legend(['Decision Boundary', 'Not admitted', 'Admitted'])
plt.show()





