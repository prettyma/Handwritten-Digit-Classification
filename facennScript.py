'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
import time
from scipy.optimize import minimize
# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    zj = 1.0 / (1.0 + np.exp(-z))
    return  zj
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    #print("inside nnObj")
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
#     print("inside nnObj2")
#     obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    #OUR FF
    trainN = training_data.shape[0]
    trainD = training_data.shape[1]
    lengthW1 = len(w1)
    lengthW2 = len(w2)
    
    training_data = np.column_stack((training_data,np.ones(trainN)))
    w1T = w1.T
    Xw1 = np.dot(training_data,w1T)
    Z = sigmoid(Xw1)
    Z = np.column_stack((Z,np.ones(Z.shape[0])))
    w2T = w2.T
    Zw2 = np.dot(Z,w2T)
    Y = sigmoid(Zw2)
    #t
    train_label=np.array(training_label)
    r = train_label.shape[0]
    rI = np.arange(r,dtype="int")
    t = np.zeros((r,2))
    t[rI,train_label.astype(int)]=1
    delL = Y - t
    #new weights
    delLT = delL.T
    new_w2 = np.dot(delLT,Z)
    delL_w2 = np.dot(delL,w2)
    Z1 = 1-Z
    ZZ = Z1*Z
    ZZ_delL_w2_T = (ZZ* delL_w2).T
    new_w1 = np.dot(ZZ_delL_w2_T,training_data) 
    
    new_w1 = np.delete(new_w1,n_hidden,0)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((new_w1.flatten(),new_w2.flatten()),0)
    obj_grad = obj_grad/trainN

    obj_val_1 = np.sum(-1*(t*np.log(Y)+(1-t)*np.log(1-Y)))
    obj_val_1 = obj_val_1/trainN
    obj_val_2 = (lambdaval/(2*trainN)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = obj_val_1 + obj_val_2
    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here
    trainN = data.shape[0]
    trainD = data.shape[1]
    lengthW1 = len(w1)
    lengthW2 = len(w2)
    
    # XX np.concatenate -> np.columnstack
    data = np.column_stack((data,np.ones(trainN)))
    w1T = w1.T
    Xw1 = np.dot(data,w1T)
    Z = sigmoid(Xw1)
    
    # XX trainN -> Z.shape(0)
    Z = np.column_stack((Z,np.ones(Z.shape[0])))
    w2T = w2.T
    Zw2 = np.dot(Z,w2T)
    Y = sigmoid(Zw2)
    
    labels = np.argmax(Y,1)
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
t_start = time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
t_end = time.time()
t_diff = t_end - t_start
print("\n Total time taken:" + str(t_diff))
pickle.dump((n_hidden,w1,w2,lambdaval),open('params.pickle','wb'))
