from numpy import *
import numpy as np

def sigmoid(x):
    return 1 / (1+exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class NeuralNet:
    def __init__(self):
        self.hiddenLayerW = None
        self.outputLayerW = None
        self.output = None
        self.MSE = None
        self.trained = False
        
    def predict( self, X ):
        ### ... YOU FILL IN THIS CODE ....
        X=array([X])
        a0=np.hstack((array([[1]*X.shape[0]]).T,X))
        #a0=X.shape[1] + 1
        print(a0)
        self.output= sigmoid(dot(sigmoid(dot(a0,self.hiddenLayerW)),self.outputLayerW))
        return self.output[0]

    def train(self,X,Y,hiddenLayerSize,epochs):    
        ## size of input layer (number of inputs plus bias)
        ni = X.shape[1] + 1
        print(ni)

        ## size of hidden layer (number of hidden nodes plus bias)
        nh = hiddenLayerSize + 1

        # size of output layer
        no = 10

        ## initialize weight matrix for hidden layer
        self.hiddenLayerW = 2*random.random((ni,nh)) - 1

        ## initialize weight matrix for output layer
        self.outputLayerW = 2*random.random((nh,no)) - 1

        ## learning rate
        alpha = 0.001

        ## Mark as not trained
        self.trained = False
        ## Set up MSE array
        self.MSE = [0]*epochs

        for epoch in range(epochs):

            ### ... YOU FILL IN THIS CODE ....
            a0=hstack((array([[1]*X.shape[0]]).T,X))  #activation of input layer
            in0=dot(a0,self.hiddenLayerW)             #input to hidden layer
            a1=sigmoid(in0)                           #activation of hidden layer
            a1[:, 0]=1                                #set bias unit for hidden layer
            in1=dot(a1,self.outputLayerW)             #input to output layer
            a2=sigmoid(in1)                           #activation of output layer
            error_out=Y-a2                            #observed error on output
            delta_out=error_out*dsigmoid(a2)            #direction of target


            ## Record MSE
            self.MSE[epoch] = mean(map(lambda x:x**2,error_out))

            ### ... YOU FILL IN THIS CODE
            error_hid=dot(delta_out,(self.outputLayerW).T) #contribution of hidden nodes to error
            delta_hid=error_hid*dsigmoid(a1)               #direction of target for hidden layer
            self.hiddenLayerW=self.hiddenLayerW+ dot(dot(alpha,a0.T),delta_hid) #hidden layer weight update
            self.outputLayerW=self.outputLayerW+dot(dot(alpha,a1.T),delta_out)  #output layer weight update

        ## Update trained flag
        self.trained = True

