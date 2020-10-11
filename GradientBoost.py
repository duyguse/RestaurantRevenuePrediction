# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:28:12 2019

@author: u21w97
"""

import numpy as np

class Node:
    def __init__(self):
        self.isLeaf = False
        self.splitFeature = None
        self.rightNode = None
        self.leftNode = None
        self.value = None
        
class Tree:
    def __init__(self):
        self.root = None
    
    def fit(self, X_train, leaf, max_depth):
        self.root = Node()
        self._fit(self.root, X_train, leaf, max_depth, 1)
        return self
    
    def calc_residual_mean(self, left_res, right_res):
        return np.mean(left_res), np.mean(right_res)
    
    def find_loss(self, X_train, y):
        y_pred = np.array([self.predict(x)for x in X_train[:,:]])
        return np.mean(np.square(y_pred - y))
        
    def calc_weight(self, X_train):
        for i in range(X_train.shape[1]):
            sorted_index = np.argsort(X_train[:,i])
            for j in range(1,X_train.shape[0]):
                return X_train[sorted_index[j],i]
                        
    def _fit(self, node, X_train, leaf, max_depth, depth):
        if depth >= max_depth:
            node.isLeaf = True
            node.value = np.mean(leaf)
        else:
            best_loss = np.inf
            best_split_feature, best_split_val = None, None
            best_left_ids, best_right_ids = None, None
            for i in range(X_train.shape[1]):
                sorted_index = np.argsort(X_train[:,i])
                for j in range(1,X_train.shape[0]-1):
                    split = X_train[sorted_index[j],i]
                    node.value = split
                    node.splitFeature = i
                    left_res = leaf[sorted_index[:j]]
                    right_res = leaf[sorted_index[j + 1:]]
                    node.leftNode = Node()
                    node.rightNode = Node()
                    node.leftNode.value, node.rightNode.value = self.calc_residual_mean(left_res, right_res)
                    node.leftNode.isLeaf = True
                    node.rightNode.isLeaf = True
                    loss = self.find_loss(X_train, leaf)
                    if loss < best_loss:
                        best_loss = loss
                        best_split_feature = i
                        best_split_val = X_train[sorted_index[j], i]
                        best_left_ids = sorted_index[:j]
                        best_right_ids = sorted_index[j + 1:] 
            if best_loss < 0.1:
                node.isLeaf = True
                node.value = np.mean(leaf)
                return
            if(best_split_feature!=None):
                node.splitFeature = best_split_feature
                node.isLeaf = False
                node.value = best_split_val
                node.leftNode = Node()
                self._fit(node.leftNode, X_train[best_left_ids], leaf[best_left_ids],
                          max_depth, depth + 1)
                node.rightNode = Node()
                self._fit(node.rightNode, X_train[best_right_ids], leaf[best_right_ids],
                          max_depth, depth + 1)
                
            else:
                node.isLeaf = True
                node.value = np.mean(leaf)              
                                                
    def predict(self, x):
        return self._predict(self.root, x)
        
    def _predict(self, node, x):
        if(node.isLeaf == True):
            return node.value   
        if x[node.splitFeature] <= node.value:
            return self._predict(node.leftNode, x)
        else:
            return self._predict(node.rightNode, x)
        
class GradientBoost:
    def __init__(self, learning_rate, depth, num_iter):
        self.learning_rate = learning_rate
        self.depth = depth
        self.num_iter = num_iter
        
    def fit(self, X_train, Y_train, X_val, Y_val, early_stopping=10):
        pred_value = np.sum(Y_train)/len(Y_train)
        self.mean = pred_value
        best_val_loss = np.inf
        trees = []
        for i in range(self.num_iter):
            print("Iteration #", i)
            residual = Y_train-pred_value
            tree = Tree().fit(X_train, residual, self.depth)
            residual_pred = np.array([tree.predict(x)for x in X_train[:,:]])
            pred_value = pred_value + self.learning_rate * residual_pred
            trees.append(tree)
            self.trees = trees
            val_loss = self.calc_loss(X_train, Y_train)
            
            print("Current loss: ", val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = i
            elif i - best_iteration >= early_stopping:
                print("Early stopping. Best Val Loss: ", best_val_loss)
                break
            
        self.best_iteration = best_iteration
        self.best_val_loss = best_val_loss
        self.tree = trees[self.best_iteration]
        self.trees = trees[:self.best_iteration+1]
            
    def calc_loss(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.square(y_pred - y))
    
    def predict(self, X):
        res = []
        for x in X:
            result = self.mean
            for tree in self.trees:
                result += self.learning_rate * tree.predict(x)
            res.append(result)
        return np.array(res)       