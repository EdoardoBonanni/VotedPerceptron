import numpy as np
import math

class VotedPerceptronC:
    #y labels in class {0, ..., 9} assume value -1,1
    #T number of epochs
    #d polynomial kernel of degree
    def __init__(self ,x, y, T, d):
        self.x = x
        self.y = y
        self.T = T
        self.d = d
        self.vIndices = []
        self.vIndices[0] = 0
        self.vValue = []
        self.vValue[0] = 0
        self.c = []
        self.c[0] = 0
        #k is the number of prediction error
        self.k = 0
        """
        #initialize value in for
        self.start = start
        #increment in for
        self.step = step
        """
    def training(self):
        for z in range(0, self.T):
            for i in range(0, len(self.x)):
                #len(v) == k+1
                #yPred is a label predict with polinomial expansion
                pred = self.predictLabelWithPolExp(self.x[i])
                temp = len(pred)
                yPred = math.copysign(1, temp)
                #yPred[lengthPred] scalar value
                if(yPred == self.y[i]):
                    #an element of c contains the number of times that prediction was good
                    self.c[self.k] += 1
                else:
                    #an element of vValue contain the value of prediction vector
                    #an element of vIndices contain the index of x(rows) with an error of prediction
                    self.vIndices.append(i)
                    nextVvalue = self.vValue[self.k] + self.y[i] * self.x[self.vIndices[i]]
                    self.vValue.append(nextVvalue)
                    #prediction vector change so c is set to 1 and number of errors increases
                    self.c.append(1)
                    self.k += 1
    def predictLabelWithPolExp(self, x):
        #pred is a vector because some partial results are important for the result of some prediction methods
        pred = []
        pred[0] = self.polynomialExpansion(self.x[0], x)
        for j in range(1, len(self.vIndices)):
            pred[j] = pred[j-1] + self.y[j] * self.polynomialExpansion(self.x[self.vIndices[j]], x)
        return pred
        #(vk*self.x) with polExp
    def predLabelPolExprl(self, x, rl):
        #pred is a vector because some partial results are important for the result of some prediction methods
        pred = []
        pred[0] = self.polynomialExpansion(self.x[0], x)
        for j in range(1, rl):
            pred[j] = pred[j-1] + self.y[j] * self.polynomialExpansion(self.x[self.vIndices[j]], x)
        return pred
    def predictLabelStandard(self, x):
        # pred is a vector because some partial results are important for the result of some prediction methods
        pred = []
        pred[0] = 0
        for j in range(1, len(self.vIndices)):
            pred[j] = pred[j-1] + self.vValue[j] * self.x[self.vIndices[j]]
        return pred
    def predLabelStandardrl(self, x, rl):
        # pred is a vector because some partial results are important for the result of some prediction methods
        pred = []
        pred[0] = 0
        for j in range(1, rl):
            pred[j] = pred[j-1] + self.vValue[j] * self.x[self.vIndices[j]]
        return pred
    def polynomialExpansion(self, a, b):
        #** exponent in python
        return (1 + np.dot(a, b)) ** self.d

    def lastUnnormalized(self, x):
        pred = self.predictLabelWithPolExp(x)
        return pred[len(pred)]
    def vote(self, x):
        predPolExp = self.predictLabelWithPolExp(x)
        pred = 0
        for i in range(1, len(self.vIndices)):
            pred += self.c[i] * np.copysign(1,predPolExp[i])
        return pred
    def avgUnnormalized(self, x):
        predPolExp = self.predictLabelWithPolExp(x)
        pred = 0
        for i in range(1, len(self.vIndices)):
            pred += self.c[i] * predPolExp[i]
        return pred
    def randomUnnormalized(self, x):
        t=0
        #len(self.vIndices) == len(self.vValue) == len(self.c)
        for i in range(1, len(self.c)):
            t += self.c[i]
        r = np.random.randint(t + 1)
        rl = 1
        sum = 0
        for i in range(0, len(self.c)):
            if(sum > r):
                break
            sum += self.c[i]
            rl += 1
        #rl largest number in (0, .., len(v)) s.t. sum in (1, rl-1) of elements of c is <= r
        pred = self.predLabelPolExprl(x, rl)
        return pred[len(pred)], rl
    def lastNormalized(self, x):
        lastUn = self.lastUnnormalized(x)
        normalPred = self.predictLabelStandard(x)
        normalizePred = np.linalg.norm(normalPred[len(normalPred)])
        if(normalizePred==0):
            return lastUn
        else:
            return lastUn/normalizePred
    def avgNormalized(self, x):
        predPolExp = self.predictLabelWithPolExp(x)
        pred = 0
        for i in range(1, len(self.vIndices)):
            normPred = self.predictLabelStandard(x)
            normalizePred = np.linalg.norm(normPred[len(normPred)])
            if(normalizePred==0):
                value = predPolExp[i]
            else:
                value = predPolExp[i]/normalizePred
            pred += self.c[i] * value
        return pred
    def randomNormalized(self, x):
        predVecrl, rl = self.randomUnnormalized(x)
        normPred = self.predLabelStandardrl(x, rl)
        normalizePred = np.linalg.norm(normPred[len(normPred)])
        if (normalizePred == 0):
            return predVecrl
        else:
            return predVecrl/normalizePred
    def methodPredictions(self, x):
        #returned value are scalar value
        lastUn = self.lastUnnormalized(x)
        vote = self.vote(x)
        avgUn = self.avgUnnormalized(x)
        randomUn = self.randomUnnormalized(x)
        lastN = self.lastUnnormalized(x)
        avgN = self.avgNormalized(x)
        randomN = self.randomNormalized(x)
        value = [lastUn, vote, avgUn, randomUn, lastN, avgN, randomN]
        return value

