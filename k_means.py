from collections import Counter

import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plot
import numpy as np
from math import dist
import random 


"""
draw_points(data):
- data is a numpy 2D array, of shape (dimensions, number), where every 
  column is a data point. The number of rows is equal to the number of 
  dimensions on the dataset. The number of columns is simply the number 
  of points in the dataset. The code can only handle 1D and 2D datasets.
  If the number of dimensions is higher, only the first two 
  dimensions are drawn.
"""
def draw_points(data):
    #print(data)
    (fig, ax) = plot.subplots()
    (dimensions, number) = data.shape
    print(data.shape)
    if (dimensions == 1):
        ax.plot(data[0,:], np.zeros((number)),'o')
    else:
        ax.plot(data[0,:], data[1,:],'o')


"""
draw_assignments(data, assignments):
- data is a numpy 2D array, of shape (dimensions, number), where every 
  column is a data point. The number of rows is equal to the number of 
  dimensions on the dataset. The number of columns is simply the number 
  of points in the dataset. The code can only handle 1D and 2D datasets.
  If the number of dimensions is higher, only the first two 
  dimensions are drawn.
- assignments is a 1D numpy array, and assignments[i] specifies the cluster
  that data[:,i] (the i-th data point) belongs to.
"""
def draw_assignments(data, assignments):
    K = np.max(assignments)
    #print(assignments)
    #print(K)
    clustering = [[] for c in range(0,K+1)]
    for c in range (0, K+1):
        clustering[c] = [data[:,i] for i in (assignments == c).nonzero()[0]]
    
    draw_clustering (clustering)

def draw_clustering(clustering):
    (fig, ax) = plot.subplots()
    number = len(clustering)
    for i in range (0, number):
        cluster = clustering[i]
        m = len(cluster)
        x = []
        y = []
        for j in range (0, m):
            v = cluster[j]
            x = x + [v[0]]
            if (len(v) > 1):
                y = y + [v[1]]
            else:
                y = y + [0]
        ax.plot(x, y,'o')

def create_dataset(pathname) :
    #theres some junk left that I'm too nervous to delete
    file = open(pathname) 
    row_arr = []
    labels = []
    for lines in file:
        #print(lines.split())
        temp_arr=[float(x) for x in lines.split()]

        row_arr.append(temp_arr)

    return row_arr

def create_test_dataset():
    data = [[1,0,1,1],[1,1,1,1],[1,0,1,0]]
    labels = [0,1,1]
    unique = [0,1]

    return data

def round_robbin(data,groups):

    clusters = [[] for _ in range(groups)]
    counter = 0
    assignment = []

    for i in range(0, len(data)):
        if (counter > groups-1):
            counter = 0

        appendee = data[i]
        #print(data)
        #print(data[i])
        #print(appendee)

        # if (len(data[i])==2):
        #     print('(%10.4f, %10.4f) --> cluster %d' % (data[i][0], data[i][1], counter+1))

        # else:
        #     print('%10.4f --> cluster %d' % (data[i][0], counter+1))

        clusters[counter].append(appendee)
        assignment.append(counter)
        #print(assignment)

        counter = counter + 1

    return assignment,clusters


def arrays_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True

def k_means(data,groups,initialize):

    #mutate the order of the dataset via shuffling
    if (initialize == 'random'):
        random.shuffle(data)
    
    #assignment is done in a round robin fashion, even for random (since it's in a random order, assignment is random)
    assignments, clusters = round_robbin(data,groups)

    data2 = copy.deepcopy(data)

    means = []
    mean_assignments = []
    temp_cluster = []
    updated = []
    MAX = 1000
    counter = 0
    data = np.array(data)

    #Initial means
    for i in  range(0, len(clusters)):

         means.append(np.array(clusters[i]).mean(axis=0).flatten())
         mean_assignments.append(i)

    # print("Manual Means")
    # print(np.array(means))
    means = np.array(means)

    #means = np.random.rand(groups,np.array(data).shape[1])
    # print("Means")
    # print(means)
    #debug means
    old_means = means.copy()

    #calculate the euclidian distance of each point from it's clusters mean
    while(True):
        old_assignments = np.array(assignments.copy())
        dist = np.linalg.norm(data - means[0,:],axis=1).reshape(-1,1)
        for i in range(1,groups):
            dist = np.append(dist,np.linalg.norm(data - means[i,:],axis=1).reshape(-1,1),axis=1)
        assignments = np.argmin(dist,axis=1)
    
        for i in set(assignments):
            means[i,:] = np.mean(data[assignments == i,:],axis=0)

        if (arrays_equal(old_assignments,assignments)):
            break
        elif(counter == MAX):
            print("Maximum runs achieved, returning current list")
            break

        counter = counter +1




    #draw_assignments(data.T,np.array(assignments))
    # print(clusters)
    # print()

    for i in range(0, len(data)):

        if (len(data[i])==2):
            print('(%10.4f, %10.4f) --> cluster %d' % (data[i][0], data[i][1], assignments[i]+1))

        else:
            print('%10.4f --> cluster %d' % (data[i][0], assignments[i]+1))

# draw_points(np.array(data).T)

data_name = sys.argv[1]
K = int(sys.argv[2])
init = sys.argv[3]

data = create_dataset(data_name)

k_means(data,K,init)
# draw_assignments(np.array(data).T, np.array(assignments))

#plot.show()