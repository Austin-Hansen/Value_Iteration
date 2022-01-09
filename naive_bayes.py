import numpy as np
from math import sqrt
from math import pi
from math import exp
import sys

def calc_prior(features, target):
	'''
	prior probability P(y)
	calculate prior probabilities
	'''
	prior = (features.groupby(target).apply(lambda x: len(x)) / self.rows).to_numpy()
	return prior

def calc_statistics(X, target):
	'''
	calculate mean, variance for each column and convert to numpy array
	'''
	means = []
	var = []
	temp = []

	#for i in range(0,2):
	temp = X[np.where(target == 1)]



	means.append(np.mean(temp,axis=1))
		#temp.append(X[np.where(target == i)])
	#var.append(np.var(temp,axis=1))

	print("Means")
	print(means)
	print("Variance")
	print(var)

	return means, var
    
def gaussian_density(class_idx, x,means,var):     
	'''
	calculate probability from gaussian density function (normally distributed)
	we will assume that probability of specific target value given specific class is normally distributed 
	probability density function derived from wikipedia:
	(1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²)), where μ is mean, σ² is variance, σ is quare root of variance (standard deviation)
	'''
	mean = means[class_idx]
	var = var[class_idx]
	numerator = np.exp((-1/2)*((x-mean)**2) / (2 * var))
	#numerator = np.exp(-((x-mean)**2 / (2 * var)))
	denominator = np.sqrt(2 * np.pi * var)
	prob = numerator / denominator
	return prob
    
def calc_posterior(x,classes,prior):
	posteriors = []

# calculate posterior probability for each class
	for i in range(len(classes)):
		prior = np.log(prior[i]) ## use the log to make it more numerically stable
		conditional = np.sum(np.log(gaussian_density(i, x))) # use the log to make it more numerically stable
		posterior = prior + conditional
		posteriors.append(posterior)
# return class with highest posterior probability
	return classes[np.argmax(posteriors)]

#create a 2D array that is similar to sklearn X or reading in a .csv file
def create_dataset(pathname) :
    #theres some junk left that I'm too nervous to delete
    file = open(pathname) 
    row_arr = []
    labels = []
    for lines in file:
        #print(lines.split())
        temp_arr=[float(x) for x in lines.split()[:-1]]
        key = lines.split()[-1]

        row_arr.append(temp_arr)
        labels.append(int(key))

    unique_labels = set(labels)
    unique_labels= list(unique_labels)

    return row_arr,labels,unique_labels

def predict(features):
	preds = [calc_posterior(f) for f in features.to_numpy()]
	return preds

def accuracy(y_test, y_pred):
	accuracy = np.sum(y_test == y_pred) / len(y_test)
	return accuracy

def naive_bayes(train,test):

	X_train,y_train,unique = create_dataset(train)

	y_train = np.array(y_train)
	X_train = np.array(X_train)

	#print(X_train[np.where(y_train == 1)])

	
	means,var = calc_statistics(X_train,y_train)
	#prior = calc_prior(X_train,y_train)

	#pred = predict(X_train)

	#print(accuracy(y_train),pred)



	#X_test,y_test,unique = create_dataset(test_data)

training_file, testing_file = sys.argv[1], sys.argv[2]

naive_bayes(training_file,testing_file)
