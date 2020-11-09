import GPy
import numpy as np
import matplotlib.pyplot as plt

def generate_random_training_data(a, b):
	X = np.array([i for i in range(a, b)]).reshape(-1, 1)
	Y = np.array([np.round(np.sin(i), 2) for i in X]).reshape(-1, 1)
	return X, Y

class Modified_GP_Regression:

	def __init__(self, X: np.array, Y: np.array):
		# GP Regression with squared exponential kernel 
		kernel = GPy.kern.RBF(input_dim=X.shape[1], )
		self.model = GPy.models.GPRegression(X, Y, kernel=kernel, initialize=True)
		self.model.optimize()
		# input space borders
		self.a = X[0]
		self.b = X[len(X)-1]

	def plot_posterior(self,):
		self.model.plot()
		plt.title("posterior")
		plt.show()

	def predict(self, x):
		return self.model.predict(x)



X, Y = generate_random_training_data(0, 10)


model = Modified_GP_Regression(X=X, Y=Y)
model.plot_posterior()