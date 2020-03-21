import VotedPerceptron as vp
import usefulFunctions as uf


def trainingClasses01(x, y, labels, T, d, start, step):
    # creation of epoch array
    # an epoch will contain all the classes of the considered epoch
    epochVIC = uf.createArray(9)
    epochCC = uf.createArray(9)
    yNewClass = []
    for l in range(0, len(labels)):
        # To handle multiclass data, we essentially reduced to 10 binary problems.
        # We trained the voted-perceptron algorithm once for each of the 10 classes.
        # When training on class l, we replaced each labeled example (xi, yi) (where yi ∈ {0,..., 9})
        # by the binary-labeled example (xi, +1) if yi = l and by (xi, −1) if yi != l (from documentation).
        print("training class", l)
        yl = []
        for i in range(0, len(y)):
            if(l == y[i]):
                yl.append(1)
            else:
                yl.append(-1)
        temp = yl.copy()
        yNewClass.append(temp)
        votedP = vp.VotedPerceptronC(x, yl, T, d, start, step)
        vIndicesContainer, cContainer = votedP.training()
        for j in range(0, len(vIndicesContainer)):
            # insert the vectors calculated in a certain epoch of class l in the container
            epochVIC[j].append(vIndicesContainer[j])
            epochCC[j].append(cContainer[j])
    return epochVIC, epochCC, yNewClass


def trainingClasses(x, y, labels, T, d, start, step):
    # creation of epoch array
    # an epoch will contain all the classes of the considered epoch
    if(d==1):
        epochVIC = uf.createArray(10)
        epochCC = uf.createArray(10)
    else:
        epochVIC = uf.createArray(30)
        epochCC = uf.createArray(30)
    yNewClass = []
    for l in range(0, len(labels)):
        # To handle multiclass data, we essentially reduced to 10 binary problems.
        # We trained the voted-perceptron algorithm once for each of the 10 classes.
        # When training on class l, we replaced each labeled example (xi, yi) (where yi ∈ {0,..., 9})
        # by the binary-labeled example (xi, +1) if yi = l and by (xi, −1) if yi != l (from documentation).
        print("training class", l)
        yl = []
        for i in range(0, len(y)):
            if(l == y[i]):
                yl.append(1)
            else:
                yl.append(-1)
        temp = yl.copy()
        yNewClass.append(temp)
        votedP = vp.VotedPerceptronC(x, yl, T, d, start, step)
        vIndicesContainer, cContainer = votedP.training()
        for j in range(0, len(vIndicesContainer)):
            # insert the vectors calculated in a certain epoch of class l in the container
            epochVIC[j].append(vIndicesContainer[j])
            epochCC[j].append(cContainer[j])
    return epochVIC, epochCC, yNewClass


def trainingC(x, y, xReduced, yReduced, labels, d):
    print("training epoch 0.1-0.9")
    VIC01, CC01, yNC01 = trainingClasses01(xReduced, yReduced, labels, 10, d, 1, 1)
    if(d == 1):
        print("training epoch 1-10")
        VIC, CC, yNC = trainingClasses(x, y, labels, 11, d, 1, 1)
    else:
        print("training epoch 1-30")
        VIC, CC, yNC = trainingClasses(x, y, labels, 31, d, 1, 1)
    return VIC01, CC01, yNC01, VIC, CC, yNC



