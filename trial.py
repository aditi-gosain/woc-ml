# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:35:11 2021

@author: aditi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

df=pd.read_csv(r"C:\Users\aditi\OneDrive\Desktop\mnist_test.csv", header=None)
m=10000
n=785
alpha = 0.00000001

X0=np.ones([m,1])
X1=(df.iloc[:, 1:n]).to_numpy()
X2 = np.hstack((X0,X1)) #(m,n)
X = np.reshape(X2,(n,m)) #(n,m)
theta=np.ones([1,n]) #(1,n)
Y1 = (df.iloc[:,0:1]).to_numpy() #(m, 1)
Y = np.reshape(Y1,(1,m)) #(1,m)

for i in range(0,30000):
    H=np.dot(theta,X) #(1,n)*(n,m)=(1,m)
    J=(1/2*m)*np.sum(np.square(H-Y))
    dJ=(1/m)*np.dot((H-Y),X2)
    theta = theta-alpha*dJ


print(J)
print(H[0][0])


        
        
    




