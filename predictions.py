import numpy as np
import usefulFunctions as uf


def lastUnnormalized(x, vIndices, xTrain, yTrain, d):
    pred = uf.predictLabelWithPolExp(x, vIndices, xTrain, yTrain, d)
    return pred[len(pred) - 1]


def vote(x, vIndices, c, xTrain, yTrain, d):
    predPolExp = uf.predictLabelWithPolExp(x, vIndices, xTrain, yTrain, d)
    pred = 0
    for i in range(1, len(vIndices)):
        pred += c[i] * np.copysign(1, predPolExp[i])
    return pred


def avgUnnormalized(x, vIndices, c, xTrain, yTrain, d):
    predPolExp = uf.predictLabelWithPolExp(x, vIndices, xTrain, yTrain, d)
    pred = 0
    for i in range(1, len(vIndices)):
        pred += c[i] * predPolExp[i]
    return pred


def randomUnnormalized(x, vIndices, c, xTrain, yTrain, d):
    t = 0
    # len(vIndices) == len(c)
    for i in range(0, len(c)):
        t += c[i]
    r = np.random.randint(t + 1)
    rl = 1
    sum = 0
    for i in range(1, len(c)):
        if (sum > r):
            break
        sum += c[i]
        rl += 1
    rl = rl - 1
    # rl largest number in (0, .., len(v)) s.t. sum in (1, rl-1) of elements of c is <= r
    pred = uf.predictLabelWithPolExp(x, vIndices, xTrain, yTrain, d)
    return pred[rl], rl


def lastNormalized(x, vIndices, xTrain, yTrain, d):
    lastUn = lastUnnormalized(x, vIndices, xTrain, yTrain, d)
    normalPred = uf.predictLabelStandard(vIndices, xTrain, yTrain)
    normalizePred = np.linalg.norm(normalPred[len(normalPred)-1])
    if (normalizePred == 0):
        return lastUn
    else:
        return lastUn / normalizePred


def avgNormalized(x, vIndices, c, xTrain, yTrain, d):
    predPolExp = uf.predictLabelWithPolExp(x, vIndices, xTrain, yTrain, d)
    pred = 0
    for i in range(1, len(vIndices)):
        normPred = uf.predictLabelStandard(vIndices, xTrain, yTrain)
        normalizePred = np.linalg.norm(normPred[len(normPred)-1])
        if (normalizePred == 0):
            value = predPolExp[i]
        else:
            value = predPolExp[i] / normalizePred
        pred += c[i] * value
    return pred


def randomNormalized(x, vIndices, c, xTrain, yTrain, d):
    predVecrl, rl = randomUnnormalized(x, vIndices, c, xTrain, yTrain, d)
    normPred = uf.predictLabelStandard(vIndices, xTrain, yTrain)
    normalizePred = np.linalg.norm(normPred[rl])
    if (normalizePred == 0):
        return predVecrl
    else:
        return predVecrl / normalizePred


def methodPredictionsGeneral(x, vIndices, c, xTrain, yTrain, d):
    # returned value of predictions are scalar value
    lastUn = lastUnnormalized(x, vIndices, xTrain, yTrain, d)
    voteValue = vote(x, vIndices, c, xTrain, yTrain, d)
    avgUn = avgUnnormalized(x, vIndices, c, xTrain, yTrain, d)
    randomUn, rl = randomUnnormalized(x, vIndices, c, xTrain, yTrain, d)
    lastN = lastUnnormalized(x, vIndices, xTrain, yTrain, d)
    avgN = avgNormalized(x, vIndices, c, xTrain, yTrain, d)
    randomN = randomNormalized(x, vIndices, c, xTrain, yTrain, d)
    value = [lastUn, voteValue, avgUn, randomUn, lastN, avgN, randomN]
    return value


def methodPredictions(x, vIndices, c, xTrain, yTrain, d):
    # returned value of predictions are scalar value
    lastUn = lastUnnormalized(x, vIndices, xTrain, yTrain, d)
    voteValue = vote(x, vIndices, c, xTrain, yTrain, d)
    avgUn = avgUnnormalized(x, vIndices, c, xTrain, yTrain, d)
    randomUn, rl = randomUnnormalized(x, vIndices, c, xTrain, yTrain, d)
    value = [lastUn, voteValue, avgUn, randomUn]
    return value

