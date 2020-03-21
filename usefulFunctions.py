import numpy as np


def createArray(l):
    array = []
    for i in range(0, l):
        arrayi = []
        array.append(arrayi)
    return array.copy()


def polynomialExpansion(a, b, d):
    # K(x,y) = (1 + x·y)**d
    # ** exponent in python
    # if we use matrices and arrays np.matmul is faster than np.dot
    # but np.matmul not permitted a product with at least one scalar value
    # (in this case we have never product with at least one scalar value)
    return (1 + np.matmul(a, b)) ** d


def predictLabelWithPolExp(x, vIndices, xTrain, yTrain, d):
    # pred is a vector because some partial results are important for the result of some prediction methods
    pred = []
    # xTrain[0] = 784 (that is the number of columns of train dataset)
    pred.append(polynomialExpansion(np.zeros(len(xTrain[0])), x, d))
    for j in range(1, len(vIndices)):
        lastPred = pred[j - 1]
        pred.append(lastPred + (yTrain[vIndices[j]] * polynomialExpansion(xTrain[vIndices[j]], x, d)))
    return pred
    #(vk*xTrain) with polExp


def predLabelPolExprl(x, rl, vIndices, xTrain, yTrain, d):
    # used for random predictions
    # pred is a vector because some partial results are important for the result of some prediction methods
    pred = []
    pred.append(polynomialExpansion(np.zeros(len(xTrain[0])), x, d))
    for j in range(1, rl):
        lastPred = pred[j - 1]
        pred.append(lastPred + (yTrain[vIndices[j]] * polynomialExpansion(xTrain[vIndices[j]], x, d)))
    return pred


def predictLabelStandard(vIndices, xTrain, yTrain):
    # pred is a vector because some partial results are important for the result of some prediction methods
    pred = []
    pred.append(0)
    for j in range(1, len(vIndices)):
        lastPred = pred[j - 1]
        pred.append(lastPred + (yTrain[vIndices[j]] * xTrain[vIndices[j]]))
    return pred


def predLabelStandardrl(rl, vIndices, xTrain, yTrain):
    # used for random predictions
    # pred is a vector because some partial results are important for the result of some prediction methods
    pred = []
    pred.append(0)
    for j in range(1, rl):
        lastPred = pred[j - 1]
        pred.append(lastPred + (yTrain[vIndices[j]] * xTrain[vIndices[j]]))
    return pred


def reduceDataset(x, y, l):
    xReduced = x[0:l].copy()
    yReduced = y[0:l].copy()
    return xReduced, yReduced

