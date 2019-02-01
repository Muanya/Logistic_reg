## Logistic Regression 

# import the necessary files
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.optimize as opt


def LrCostFunc(Theta, inputs, labels, lamda=0):
	m, n = inputs.shape
	Theta = np.array(Theta)[np.newaxis].T


	h = np.dot(X, Theta)
	h = sigmoid(h)
	reg_theta = Theta
	reg_theta[0] = 0
	error = (labels * np.log(h) + (1 - labels)*np.log(1 - h)) / -m
	J =  np.sum(error) + (0.5* lamda)/m * np.sum(reg_theta**2)

	# calculating the derivative
	dJ_dT = (np.dot(inputs.T, (h - labels)) + lamda * reg_theta.reshape(3, 1)) / m

	return J, dJ_dT



def sigmoid(x):
	return 1/(1 + np.exp(-x))

def trainLr(X, y, r_lambda=0, iterations=12500, lr=0.00019):
	ini_theta = np.zeros(3, dtype='int')

	print('\n Executing optimization........\n')

	result = opt.minimize(LrCostFunc, ini_theta, args=(X,y), method='TNC',jac=True, options={'maxiter':400})

	# result = opt.fmin_tnc(func = LrCostFunc, x0 = ini_theta, args=(X, y))

	return (result)


data = pd.read_csv('data.txt', header=None)
data.columns = ['test', 'exam', 'result']

X = np.c_[data['test'], data['exam']]
m = X.shape[0]
X = np.c_[np.ones((m, 1)), X]
y  = np.c_[data['result']]


[m,n] = np.shape(X) #(100,2)
initial_theta = np.zeros((n), dtype=int)
print(initial_theta)





[cost, grad] = LrCostFunc(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n',grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

test_theta = [-24, 0.2, 0.2]
#test_theta_value = np.array([-24, 0.2, 0.2])[np.newaxis]  #This command is used to create the 1D row array 

#test_theta = np.transpose(test_theta_value) # Transpose 
#test_theta = test_theta_value.transpose()
[cost, grad] = LrCostFunc(test_theta, X, y)

print('\nCost at test theta: \n', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n',grad);
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')


Theta = trainLr(X, y)

print('=================================')
print('Cost at theta found : \n', Theta.fun);
print('\nExpected cost (approx): 0.203\n');
print('\ntheta: \n',Theta.x);
print('\nExpected theta (approx):\n');
print(' -25.161\n 0.206\n 0.201\n');


