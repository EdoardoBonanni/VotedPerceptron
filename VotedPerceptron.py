import numpy as np
import math

class VotedPerceptron:
    #T number of epochs
    #d polynomial kernel of degree
    def __init__(self ,x, y, T, d):
        self.x = x
        self.y = y
        self.T = T
        self.d = d
        self.w = 0
        self.vTrain = []
        self.vTrain[0] = 0
        self.vLabel = []
        self.vLabel[0] = 0
        self.c = []
        self.c[0] = 0
        self.nErr = 0

def training(self):
    k=0
    for z in range(0, self.T):
        for i in range(0, self.x.getRows()):
            yPred = math.copysign(1, predictionVectorWithPolExp(self.x[i]))
            if(yPred == self.y[i]):
                self.c[k] += 1
            else:
                self.vTrain.append(i)
                self.vLabel.append(self.y[i])
                self.c.append(1)
                k += 1
                self.nErr += 1

def predictionVectorWithPolExp(self, x):
    pred = []
    pred[0] = polynomialExpansion(self.x.getColumns(), x)
    for j in range(1, len(self.vTrain)):
        pred[j] = pred[j-1] + self.vLabel[j] * polynomialExpansion(self.x[self.vTrain[j]], x)
    return pred
    #(vk*x)
"""
def predictionVectorStandard(self, x):
    pred = []
    pred[0] = 0
    for j in range(1, len(self.vTrain)):
        pred[j] = pred[j-1] + self.vLabel[j] * self.x[self.vTrain[j]]
    return pred
"""
def polynomialExpansion(self, xi, xj):
    return (1 + np.dot(xi, xj)) ** self.d

def lastUnnormalized(self, x):
    return predictionVectorWithPolExp(x)
def vote(self, x):
    otherPred = predictionVectorWithPolExp(x)
    pred = 0
    for i in range(1, len(self.vTrain)):
        pred += self.c[i] * np.copysign(1,otherPred[i])
    return pred
def avg_unnormalized(self, x):
    otherPred = predictionVectorWithPolExp(x)
    pred = 0
    for i in range(1, len(self.vTrain)):
        pred += self.c[i] * otherPred[i]
    return pred
