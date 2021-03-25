# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:35:11 2021

@author: aditi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

df=pd.read_csv(r"C:\Users\aditi\OneDrive\Desktop\mnist_train_small.csv", header=None)
m=df.shape[0]
n=df.shape[1]
alpha = 0.000009
iteration = 10000

X0=np.ones([m,1])
X1=(df.iloc[:, 1:n]).to_numpy()
X2 = np.hstack((X0,X1)) #(m,n)
Xmean = (np.sum(X2, axis=0))/m #(1,n)
Xnorm=(X2-Xmean)/255 #(m,n)-(1,n)
X = (Xnorm.T) #(n,m)
theta=np.zeros([1,n]) #(1,n)
Y1 = (df.iloc[:,0:1]).to_numpy() #(m, 1)
Ymean=(np.sum(Y1, axis=0))/m
Ynorm=(Y1-Ymean)/9
Y = (Ynorm.T) #(1,m) 


"""

X0=np.ones([m,1])
X1=(df.iloc[:, 1:n]).to_numpy()
X2 = np.hstack((X0,X1)) #(m,n)
X = X2.T #(n,m)
theta=np.ones([1,n]) #(1,n)
Y1 = (df.iloc[:,0:1]).to_numpy() #(m, 1)
Y = Y1.T #(1,m) 
"""

for i in range(iteration):
    H=np.dot(theta,X) #(1,n)*(n,m)=(1,m)
    J=(1/2*m)*np.sum(np.square(H-Y))
    dJ=(1/m)*np.dot((H-Y),X2)
    theta = theta-alpha*dJ
    pl.scatter(i,J)


print(J)
print(H[0][0])
print("alpha: ",alpha)
print("no of iterations: ",iteration)
print(np.corrcoef(Y,H))


"""
train:
20500370.018379975
-0.0016544027488194658
alpha:  1e-09
no of iterations:  10000
[[1.        0.6719779]
 [0.6719779 1.       ]]

test:
5132562.971453594
0.0018908526333176138
alpha:  1e-09
no of iterations:  10000
[[1.         0.68606294]
 [0.68606294 1.        ]]

train:
20096089.860908046
-0.005890091101307058
alpha:  3e-09
no of iterations:  12000
[[1.         0.67258053]
 [0.67258053 1.        ]]

train
19066613.994265143
-0.017138565264805138
alpha:  9e-09
no of iterations:  12000
[[1.        0.6742219]
 [0.6742219 1.       ]]

train:
 18907049.089846704
-0.018946768161832446
alpha:  1e-08
no of iterations:  12000
[[1.         0.67449161]
 [0.67449161 1.        ]]

train:
16309028.183460522
-0.05147254125310481
alpha:  3e-08
no of iterations:  12000
[[1.         0.67966565]
 [0.67966565 1.        ]]

train:
12512612.080896787
-0.11708041856658775
alpha:  9e-08
no of iterations:  12000
[[1.         0.69300233]
 [0.69300233 1.        ]]

train: 
 12186971.321934313
-0.12457006081913656
alpha:  1e-07
no of iterations:  12000
[[1.         0.69495188]
 [0.69495188 1.        ]]   

train:
 9054555.546721786
-0.153698184689001
alpha:  9e-07
no of iterations:  12000
[[1.         0.75037627]
 [0.75037627 1.        ]]

train:
8991969.979664465
-0.1503628829340999
alpha:  1e-06
no of iterations:  12000
[[1.        0.7522757]
 [0.7522757 1.       ]]


train:
8156116.748056753
-0.16334916096922392
alpha:  9e-06
no of iterations:  12000
[[1.         0.77807468]
 [0.77807468 1.        ]]

test:
1906578.738010733
0.15711864123471098
alpha:  9e-06
no of iterations:  12000
[[1.         0.79487168]
 [0.79487168 1.        ]]

test:
1919627.1496451371
0.1532459874119466
alpha:  9e-06
no of iterations:  10000
[[1.         0.79329404]
 [0.79329404 1.        ]]


train:
8204481.754946703
-0.16125228016212576
alpha:  9e-06
no of iterations:  10000
[[1.         0.77658353]
 [0.77658353 1.        ]]

"""


        
        
    




