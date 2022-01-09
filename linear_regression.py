import numpy as np
from math import sqrt
from math import pi
from math import exp
import scipy
import sys

#create a 2D array that is similar to sklearn X or reading in a .csv file
def create_dataset(pathname,degrees) :
	#theres some junk left that I'm too nervous to delete
	file = open(pathname) 
	row_arr = []
	labels = []
	for lines in file:
		#print(lines.split())
		temp_arr=[float(x) for x in lines.split()[:-1]]
		key = float(lines.split()[-1])
		#print(len(temp_arr))
		row_arr.append(temp_arr)
		labels.append(key)
	return row_arr,labels


def basis(X,degrees):

	#print(X)
	#X=[[2,2,2],[2,2,2]]
	phi_temp = []
	phi = []
	#print(len(X))
	#N traverses column (down row)
	for M in range(len(X)):
		#M should traverse row
		phi_temp.append(1)
		for N in range(0,len(X[0])):
			for i in range(1,degrees+1):
				phi_temp.append(X[M][N]**i)
		phi.append(phi_temp)
		phi_temp=[]

	#phi = [x**degrees for x in X]

	#print(np.array(phi))

	return np.array(phi)

def little_phi(X,degrees):

	lp=[]
	phi_temp=[]

	for M in range(len(X)):
	#M should traverse row
		phi_temp.append(1)
		for N in range(0,len(X[0])):
			for i in range(1,degrees+1):
				phi_temp.append(X[M][N]**i)
		lp.append(phi_temp)
		phi_temp = []

	#print("little phi")
	#print(lp)
	return lp

# Define a functin for our own implementation.
def get_regression_coefs(X, y,lam):
    """Takes a feature matrix and adds 1s as intercept, and then
    apply partial derivative on the squared errors
    """
    w=[]
    #y=[2,2]
    w.append(normal_equation(X,y,lam))

    return w

#def calculate_output()

#normal equation function
def normal_equation(x,y,lam):

	M=len((np.dot(x.transpose(), x)))
	#print("length of x*xt")
	#print(M)

    # calculate weight vector with the formula inverse of(x.T* x)*x.T*y
	z = np.linalg.pinv(((lam*np.identity(M))+(np.dot(x.transpose(), x))))
	w = np.dot(np.dot(z, x.transpose()), y)
	return w

def get_value(w,lp,y):

	w=np.array(w)
	wt=w.transpose()

	for i in range(0,len(lp)):
		#print(lp[i])
		est_value = np.dot(w,np.array(lp[i]))
		#print(est_value)
		#print("ID%5d, output=%14.4f, target value = %10.4f, squared error = %.4f" % (i+1,est_value,y[i],square_error(est_value,y[i])))

def square_error(est,act):
	#print(est)
	#print(act)

	error = pow((est[0]-act),2)

	return error


def linear_regression(training_data, test_data, degrees, lam):

	data,labels = create_dataset(training_data,degrees)
	data = np.array(data)#.flatten()
	labels= np.array(labels)
	weights = []
	#print(data)

	lf = little_phi(data,degrees)

	#print(labels)
	basis_arr = basis(data,degrees)
	#print(basis_arr)
	#print("Basis array")
	#print(basis_arr)
	#print(np.polyfit(data,labels,3))

	#basis_arr = normalize(basis_arr)


	weights = get_regression_coefs(basis_arr,labels,lam)

	#due to some weirdness, weights is a list with one np array in it
	for i in range(len(weights[0])):
		print("w%d=%.4f" % (i,weights[0][i]))

	#testing portion of the code, has been decently accurate on sets that it was tested on
	data,labels = create_dataset(test_data,degrees)
	lf = little_phi(data,degrees)



	get_value(weights,lf,labels)



	#print(weights)
	#for i in range(0,len(weights)):
		#print(weights[i])
	#print("printing data")
	#print(data)
	#print(labels)

linear_regression("pendigits_training.txt", "pendigits_test.txt", 2, 1)
