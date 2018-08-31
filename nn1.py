import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sympy as sp
import scipy as sc
from sklearn.model_selection import train_test_split
import time


def sigmoid(z):
    a=1/(1+np.exp(-z))
    return a
def hypothesis(theta,x):
    H= np.dot(theta,x)
    return H
#print(sigmoid(0))

dataset= np.genfromtxt("data_2genre.csv",delimiter=",")
#print(dataset)

dataset=dataset[1:,]
x= (dataset[0:,1:-1])/1000
#print(x.shape)
y_raw=dataset[:,-1]

y=np.zeros((200,1))
for i in range(200):
    y[i]= y_raw[i]

m=len(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
#print(x_train.shape[1])

theta1= np.random.rand(x_train.shape[1],10)
theta2= np.random.rand(10,5)
theta3=np.random.rand(5,1)
#print(theta1.shape,theta2.shape)
layer2= sigmoid(np.dot(x_train,theta1))
layer3= sigmoid(np.dot(layer2,theta2))
layer4= sigmoid(np.dot(layer3,theta3))    #layer 3 is the #1 output layer  
#print(theta1.shape,theta2.shape)
#print(layer3.shape)

delL=layer4-y_train
#print(delL.shape)
del3=(np.dot(delL,np.transpose(theta3)))*(layer3*(1-layer3))
del2=(np.dot(del3,np.transpose(theta2)))*(layer2*(1-layer2))
'''
J3= np.dot(np.transpose(delL),layer3)
J2= np.dot(np.transpose(del3),layer2)
J1= np.dot(np.transpose(del2),x_train)
J= np.zeros((len,1))
for i in range(J3.shape):
    J[i]=J3[i]
'''
alpha=0.01
nditer=3000

for i in range(nditer):
    thetha3=theta3-alpha*(1/len(layer3))*np.dot(np.transpose(delL),layer3)
    thetha2=theta2-alpha*(1/len(layer2))*np.dot(np.transpose(layer2),del3)
    thetha1=theta1-alpha*(1/len(x_train))*np.dot(np.transpose(x_train),del2)
#print(theta3,theta2,theta1)



op1=sigmoid(np.dot(x_test,theta1))
op2=sigmoid(np.dot(op1,theta2))           
op=sigmoid(np.dot(op2,theta3))    
c=0
for i in range(op.shape[0]):
    if op[i][0]>0.7:
       op[i][0]=1;
    else:
        op[i][0]=0;
    if op[i]==y_test[i]:
       c=c+1;
print(c/op.shape[0])
        
