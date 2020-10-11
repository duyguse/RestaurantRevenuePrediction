# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:28:12 2019

@author: u21w97
"""

import numpy as np
import pandas as pd 

class Preprocess:
    def __init__(self, train):
        self.x_train = train
        
    def Age(self, train_new):         
        now = pd.Timestamp('now')
        self.x_train['dob'] = pd.to_datetime(self.x_train['Open Date'], format='%m/%d/%Y')    
        train_new['age'] = (now - self.x_train['dob']).astype('<m8[Y]')  
        return train_new   
        
    def P_Groups(self, train_new, train_check): 
        scaled = []
        for i in range(1,38):
            scaled.append("P" + str(i))     
        if(train_check):
            scaled.append("revenue")    
        for column in self.x_train[scaled]: 
            new_column = []   
            i=0            
            for data in self.x_train[column]:
                if(int(data) == 0):
                    distance = []  
                    row_num = []
                    filtered = self.x_train[self.x_train['City Group'] == self.x_train['City Group'][i]][self.x_train['Type'] == self.x_train['Type'][i]][scaled]                    
                    for index, row in filtered.iterrows():
                        distance.append(np.sqrt(sum((row-self.x_train[scaled].ix[i])**2)))
                        row_num.append(index)                        
                    m2, mini = self.second_smallest(distance,min(distance))
                    mini2=float('inf')                    
                    while(self.x_train[column][row_num[mini]] == 0 and mini2 != mini ):
                        mini2 = mini
                        m2, mini = self.second_smallest(distance, m2)
                    new_column.append(self.x_train[column][row_num[mini]])                                       
                elif(int(data) >= 10):
                    new_column.append(np.sqrt(data))                    
                else:
                    new_column.append(data)
                i=i+1    
            train_new[column] = new_column 
        return train_new
        
    def City_Group(self, train_new): 
        new_column = []
        for data in self.x_train['City Group']:
            if(data == 'Big Cities'):
                new_column.append(1)
            else:
                new_column.append(2)   
                
        train_new['City Group'] = new_column
        return train_new
        
    def Type(self, train_new):
        new_column = []
        
        for data in self.x_train['Type']:
            if(data == 'DT'):
                new_column.append(1)
            elif(data == 'FC'):
                new_column.append(2)
            elif(data == 'IL'):
                new_column.append(3)
            else:
                new_column.append(4)                
        train_new['Type'] = new_column
        return train_new
    
    def City(self, train_new):
        train_new['City'] = self.x_train['City']
        return train_new    
    
    def second_smallest(self, numbers, minimum):
         m1 = float('inf')
         mini = 0
         i=0
         for x in numbers:
             if minimum < x and x < m1:
                 m1 = x
                 mini = i             
             i=i+1
         return m1, mini
     