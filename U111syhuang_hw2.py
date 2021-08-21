#!/usr/bin/env python
# coding: utf-8



import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input",dest = "w",type = float)
parser.add_argument("--training_times",dest = "training_times",type = int)
parser.add_argument("--learning_rate",dest = "learning_rate",type = float)
args = parser.parse_args()



#print ("w: ",args.w,"learning_rate: ",args.learning_rate,"training_times: ",args.training_times)

X_training = torch.tensor([1.0,2.0,3.0])
Y_training = torch.tensor([2.0,4.0,6.0])

w = torch.tensor(args.w)
w.requires_grad = True
#print (w.requires_grad)

for i in range(args.training_times):
    #w_grad = 0
    #loss = Y_training[j] - (X_training[j]*w)
    #total loss += (Y_training[j] - (X_training[j]*w))^2
    #for j in range(len(X_training)):
        #w_grad =  w_grad - 2.0*(Y_training[j] - (X_training[j]*args.w))*X_training[j]
    y = X_training*w
    loss = (Y_training - y)**2
    total_loss = loss.sum()
    total_loss.backward()
    
    with torch.no_grad():
      w.data = w.data - args.learning_rate*w.grad.data
    w.grad.zero_()
    

print (w.data)






