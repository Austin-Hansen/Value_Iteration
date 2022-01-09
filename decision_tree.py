from collections import Counter

import numpy as np

import os
import sys

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
        labels.append(key)

    uniquentropy_leftabels = set(labels)
    uniquentropy_leftabels= list(uniquentropy_leftabels)

    return np.array(row_arr),np.array(labels),np.array(uniquentropy_leftabels).T

def create_test_dataset():
    data = [[1,0,1,1],[1,1,1,1],[1,0,1,0]]
    labels = [0,1,1]
    unique = [0,1]

    return np.array(data),np.array(labels),np.array(unique).T

class Node:
    def __init__(
        self, attribute=None, threshold=None, left=None, right=None, value=None,gain = 0.0):
        self.attribute = attribute
        self.threshold = threshold
        self.left_child = left
        self.right_child = right
        #I based this node design after an old assignment I did, and somehow I conflated value and attribute while making this, almost entirely broke it
        self.value = value
        self.distribution = None
        self.gain = gain
        self.entropy = 0.0

    def is_leaf(self):
        if(self.threshold==-1):
            return True
        else:
            return False


class DecisionTree:
    def __init__(self, samples_split=2, n_feats=None,unique=0,random=True,pruning=50):
        self.samples_split = samples_split
        self.n_feats = n_feats
        self.root = None
        self.pruning_thr = pruning
        self.num_nodes = 0
        self.unique = unique
        self.randomized = random

    def DTL_TopLevel(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        default = self.distribution(y)
        self.root = self.DTL(X, y,default)

    def predict(self, X,dist_flag=0):
        #print("dist flag")
        #print(dist_flag)
        if (dist_flag==0):
            return np.array([self.navigate_tree(x, self.root) for x in X])
        else:
            prediction = np.array([self.navigate_tree(x, self.root) for x in X])
            distributions = np.array([self.navigate_tree(x, self.root,dist_flag=1) for x in X])
            return (prediction,distributions)

    def DTL(self, X, y, default):
        samples, attributes = X.shape
        num_labels = len(np.unique(y))
        # stopping criteria, make a leaf node
        if(samples < self.pruning_thr):
            node = Node(value= np.where(default == np.amax(default))[0][0])
            node.distribution = default
            node.gain = 0.0
            node.threshold = -1
            return node
     
        if (num_labels == 1):
            leaf_value = self.common_label(y)
            leaf_value = int(np.where(default == np.amax(default))[0][0])
            self.num_nodes = self.num_nodes+1
            node = Node(value=leaf_value)
            node.distribution = default
            node.threshold = -1
            node.gain = 0.0
            return node

        #create a randomized attribute, can lower the computation time by a minor degree
        rand_att = np.random.choice(attributes, self.n_feats, replace=False)

        # choose the best attribute
        if(self.randomized == False):
            best_feat, best_thresh,best_gain = self.choose_attribute(X, y, rand_att)
        else:
            best_feat, best_thresh,best_gain = self.randomize(X, y, rand_att)

        # grow the children that result from the split
        left_splits, right_splits = self.split(X[:, best_feat], best_thresh)
        start_dist = self.distribution(y)
        left = self.DTL(X[left_splits, :], y[left_splits], start_dist)
        right = self.DTL(X[right_splits, :], y[right_splits], start_dist)
        self.num_nodes=self.num_nodes+1
        return Node(best_feat, best_thresh, left, right,value=best_feat,gain=best_gain)

    def choose_attribute(self, X, y, attributes):
        best_gain = -1
        split_idx, split_thresh = None, None
        for A in attributes:
            X_column = np.array(X[:, A])
            L = np.amin(X_column)
            M = np.amax(X_column)
            temp = np.arange(1,51)*((M-L)/51)
            temp = temp+L
            thresholds = temp
            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_att = A
                    best_thresh = threshold

        return best_att, best_thresh, best_gain

    def randomize(self, X, y, attributes):
        best_gain = -1
        split_idx, split_thresh = None, None
        A = np.random.randint(0,max(attributes))
            
        X_column = np.array(X[:, A])
        L = np.amin(X_column)
        M = np.amax(X_column)
        temp = np.arange(1,51)*((M-L)/51)
        temp = temp+L
        thresholds = temp
        for threshold in thresholds:
            gain = self.information_gain(y, X_column, threshold)

            if gain > best_gain:
                best_gain = gain
                best_thresh = threshold

        return A, best_thresh, best_gain


    def information_gain(self, y, X_column, split_thresh):
        # parent's loss
        parent_entropy = self.entropy(y)

        # generate split
        left_splits, right_splits = self.split(X_column, split_thresh)

        if len(left_splits) == 0 or len(right_splits) == 0:
            return 0

        # average weighted loss of child nodes
        n = len(y)
        num_left, num_right = len(left_splits), len(right_splits)
        entropy_left, entropy_right = self.entropy(y[left_splits]), self.entropy(y[right_splits])
        child_entropy = (num_left / n) * entropy_left + (num_right / n) * entropy_right

        # information gain = h(parent) - h(children)
        info_gain = parent_entropy - child_entropy
        return info_gain

    def split(self, X_column, split_thresh):
        left_splits = np.argwhere(X_column < split_thresh).flatten()
        right_splits = np.argwhere(X_column >= split_thresh).flatten()
        return left_splits, right_splits

    def entropy(self,y):
        temp,counts = np.unique(y,return_counts=True)
        ps = counts/counts.sum()
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def navigate_tree(self,X,node,dist_flag=0):

        if node.is_leaf():
            if(dist_flag==0):
                return node.value
            else:
                return node.distribution
        elif (X[node.attribute]<node.threshold):
            return self.navigate_tree(X,node.left_child,dist_flag)
        
        return self.navigate_tree(X,node.right_child,dist_flag)

    def common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def distribution(self,labels):
    
        num_labels = len(self.unique)
        count = np.zeros(num_labels)

        for i in range(len(labels)):
            count[int(labels[i])] += 1

        return np.array(count)/len(labels)

#return an array of trained decision trees
def tree_trainer(X,y,num,labels,randomized,pruning_thr):

    forest = []
    for i in range(0,num):
        clf = DecisionTree(unique=labels,random=randomized,pruning=pruning_thr)
        clf.DTL_TopLevel(X, y)
        forest.append(clf)
    return forest
def calc_accuracy(y_true, y_pred):

    correct = np.sum(y_true == y_pred)
    accuracy = correct/ len(y_true)
    return accuracy

def get_height(tree):
    if tree is None:
        return 0
    else:
        left_child = get_height(tree.left_child)
        right_child = get_height(tree.right_child)
        if left_child > right_child:
            return left_child + 1
        else:
            return right_child+1
#global variable is not the best solution 
node_num = 1
def print_forest(tree, height, tree_num):
    if tree is None:
        return
    if height == 1:
        global node_num
        print("tree=" + "%2d"%int(tree_num) + ", node=" + "%3d"%int(node_num) + ", feature=" + "%2d"%tree.value + ", thr=" + "%6.2f"%tree.threshold + ", gain=" + "%f"%tree.gain)
        node_num = node_num+1
    elif height > 1:
        print_forest(tree.left_child, height-1, tree_num)
        print_forest(tree.right_child, height-1, tree_num)

def decision_tree(training_data,test_data,option,pruning_thr):

    num_trees = 1
    random = True

    if (option == 'forest3'):
        num_trees = 3
    elif (option == 'forest15'):
        num_trees = 15
    elif (option == 'optimized'):
        random = False


    X_train,y_train,unique = create_dataset(training_data)
    X_test,y_test,unique = create_dataset(test_data)

    forest = tree_trainer(X_train,y_train,num_trees,unique,random,pruning_thr)

    y_pred = []
    tree_dist = []

    for i in range(len(forest)):
        height = get_height(forest[i].root)
        global node_num
        node_num = 1

        for j in range(1, height+1):
            print_forest(forest[i].root, j, i+1)

    for tree in forest:
        temp = tree.predict(X_test,dist_flag=1)
        y_pred.append(temp[0])
        tree_dist.append(temp[1])


    temp = tree_dist[0]

    for i in range(1,len(tree_dist)-1):
        temp = temp+tree_dist[i]

    temp=temp/3
    consensus = []

    accuracy = []
    #calculate accuracy 
    for i in range(0,len(temp)):
        preconsensus = np.where(temp[i] == np.amax(temp[i]))[0]
        if (len(preconsensus)>1):
            consensus.append(str(np.random.choice(preconsensus)))
            if (consensus[-1]==y_test[i]):
                accuracy.append(1/len(preconsensus))
            else:
                accuracy.append(0.0)
        else:
            consensus.append(str(preconsensus[0]))
            if (consensus[-1]==y_test[i]):
                accuracy.append(1.0)
            else:
                accuracy.append(0.0)
        print("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f" % (i+1,int(consensus[-1]),int(y_test[i]),accuracy[i]))

    acc=sum(accuracy)/len(temp)
    print("classification accuracy=%6.4f\n" % acc)

    #reliably check the accuracy calculation
    #acc = calc_accuracy(np.array(y_test), np.array(consensus))
    #print(clf.num_nodes)

    #print("Accuracy:", acc)

#test_decision_tree
#decision_tree("pendigits_training.txt","pendigits_test.txt","optimized",50)
#yields accuracy = 0.8382
training_data = sys.argv[1]
test_data = sys.argv[2]
option = sys.argv[3]
pruning_thr = int(sys.argv[4])
decision_tree(training_data,test_data,option,pruning_thr)
