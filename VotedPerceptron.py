import numpy as np
import usefulFunctions as uf

class VotedPerceptronC:
    # y: labels in class {0, ..., 9} assume value -1,1
    # T: number of epochs
    # d: polynomial kernel of degree
    # start: initialize value in for
    # step: increment in for
    def __init__(self, x, y, T, d, start, step):
        self.x = x
        self.y = y
        self.T = T
        self.d = d
        self.vIndices = []
        self.vIndices.append(0)
        self.c = []
        self.c.append(0)
        # k is the number of prediction error
        self.k = 0
        self.start = start
        self.step = step

    def training(self):
        vIndicesContainer = []
        cContainer = []
        for z in range(self.start, self.T, self.step):
            print("training epoch")
            for i in range(0, len(self.x)):
                # len(v) == k+1
                # pred: label predict with polinomial expansion
                pred = self.predict(self.x[i])
                temp = pred[len(pred) - 1]
                yPred = np.copysign(1, temp)
                if(yPred == self.y[i]):
                    # an element of c contains the number of times that prediction was good
                    self.c[self.k] += 1
                else:
                    # an element of vIndices contain the index of x(rows) and y with an error of prediction
                    self.vIndices.append(i)
                    # prediction vector change so c is set to 1 and number of errors increases
                    self.c.append(1)
                    self.k += 1
            # need to copy the vector through a copy (deepcopy)
            # because otherwise after append the vector prediction (vIndices) and c will reset
            tempV = self.vIndices.copy()
            tempC = self.c.copy()
            vIndicesContainer.append(tempV)
            cContainer.append(tempC)
        return vIndicesContainer, cContainer

    def predict(self, x):
        # pred is a vector because some partial results are important for the result of some prediction methods
        pred = uf.predictLabelWithPolExp(x, self.vIndices, self.x, self.y, self.d)
        return pred
        # (vk*xTrain) with polExp



