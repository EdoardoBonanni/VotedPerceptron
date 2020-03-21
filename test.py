import numpy as np
import matplotlib.pyplot as plt
import predictions as p
from matplotlib.ticker import ScalarFormatter


def prepareTest01(xTrain01, yNewTrain01, VIC01, CC01, xTest, yTest, d):
    err = []
    for i in range(0, len(VIC01)):
        print('test epoch 0,', i+1, sep='')
        errlastUn = 0
        errvote = 0
        erravgUn = 0
        errrandomUn = 0
        for x in range(0, len(xTest)):
            print("test istance", x)
            lastUn = []
            vote = []
            avgUn = []
            randomUn = []
            maxlastUn = 0
            maxvote = 0
            maxavgUn = 0
            maxrandomUn = 0
            for j in range(0, len(VIC01[i])):
                print("test class", j)
                #value of lastUn, vote, ... of a specific class l
                value = p.methodPredictions(xTest[x], VIC01[i][j], CC01[i][j], xTrain01, yNewTrain01[j], d)
                lastUn.append(value[0])
                vote.append(value[1])
                avgUn.append(value[2])
                randomUn.append(value[3])
            #search the highest prediction value of all classes (the highest value of lastUn, Vote, ....)
            maxlastUn = np.argmax(lastUn)
            maxvote = np.argmax(vote)
            maxavgUn = np.argmax(avgUn)
            maxrandomUn = np.argmax(randomUn)
            # sum 1 if the prediction is false
            if (maxlastUn != yTest[x]):
                errlastUn += 1
            if (maxvote != yTest[x]):
                errvote += 1
            if (maxavgUn != yTest[x]):
                erravgUn += 1
            if (maxrandomUn != yTest[x]):
                errrandomUn += 1
        #error rate in percent committed in a certain epoch
        errlastUnPerc = (errlastUn / len(yTest)) * 100
        errvotePerc = (errvote / len(yTest)) * 100
        erravgUnPerc = (erravgUn / len(yTest)) * 100
        errrandomUnPerc = (errrandomUn / len(yTest)) * 100
        errEpoch = [errlastUnPerc, errvotePerc, erravgUnPerc, errrandomUnPerc]
        err.append(errEpoch)
    #err contains the set of error committed at each epoch
    return err


def prepareTest(xTrain, yNewTrain, VIC, CC, xTest, yTest, d):
    err = []
    for i in range(0, len(VIC)):
        print("test epoch", i+1)
        errlastUn = 0
        errvote = 0
        erravgUn = 0
        errrandomUn = 0
        for x in range(0, len(xTest)):
            print("test istance ", x)
            lastUn = []
            vote = []
            avgUn = []
            randomUn = []
            maxlastUn = 0
            maxvote = 0
            maxavgUn = 0
            maxrandomUn = 0
            for j in range(0, len(VIC[i])):
                print("test class", j)
                #value of lastUn, vote, ... of a specific class l
                value = p.methodPredictions(xTest[x], VIC[i][j], CC[i][j], xTrain, yNewTrain[j], d)
                lastUn.append(value[0])
                vote.append(value[1])
                avgUn.append(value[2])
                randomUn.append(value[3])
            #search the highest prediction value of all classes (the highest value of lastUn, Vote, ....)
            maxlastUn = np.argmax(lastUn)
            maxvote = np.argmax(vote)
            maxavgUn = np.argmax(avgUn)
            maxrandomUn = np.argmax(randomUn)
            # sum 1 if the prediction is false
            if (maxlastUn != yTest[x]):
                errlastUn += 1
            if (maxvote != yTest[x]):
                errvote += 1
            if (maxavgUn != yTest[x]):
                erravgUn += 1
            if (maxrandomUn != yTest[x]):
                errrandomUn += 1
        #error rate in percent committed in a certain epoch
        errlastUnPerc = (errlastUn / len(yTest)) * 100
        errvotePerc = (errvote / len(yTest)) * 100
        erravgUnPerc = (erravgUn / len(yTest)) * 100
        errrandomUnPerc = (errrandomUn / len(yTest)) * 100
        errEpoch = [errlastUnPerc, errvotePerc, erravgUnPerc, errrandomUnPerc]
        err.append(errEpoch)
    #err contains the set of error committed at each epoch
    return err


def prepareTestComplete01(xTrain01, yNewTrain01, VIC01, CC01, xTest, yTest, d):
    # not implemented result
    err = []
    for i in range(0, len(VIC01)):
        print("test epoch 0,", i+1)
        errlastUn = 0
        errvote = 0
        erravgUn = 0
        errrandomUn = 0
        errlastN = 0
        erravgN = 0
        errrandomN = 0
        for x in range(0, len(xTest)):
            print("test", x)
            lastUn = []
            vote = []
            avgUn = []
            randomUn = []
            lastN = []
            avgN = []
            randomN = []
            for j in range(0, len(VIC01[i])):
                #value of lastUn, vote, ... of a specific class l
                value = p.methodPredictionsGeneral(xTest[x], VIC01[i][j], CC01[i][j], xTrain01, yNewTrain01[j], d)
                lastUn.append(value[0])
                vote.append(value[1])
                avgUn.append(value[2])
                randomUn.append(value[3])
                lastN.append(value[4])
                avgN.append(value[5])
                randomN.append(value[6])
            #search the highest prediction value of all classes (the highest value of lastUn, Vote, ....)
            maxlastUn = np.argmax(lastUn)
            maxvote = np.argmax(vote)
            maxavgUn = np.argmax(avgUn)
            maxrandomUn = np.argmax(randomUn)
            maxlastN = np.argmax(lastN)
            maxavgN = np.argmax(avgN)
            maxrandomN = np.argmax(randomN)
            #sum 1 if the prediction is false
            errlastUn = ((errlastUn + 1) if (maxlastUn!=yTest[x]) else errlastUn)
            errvote = ((errvote + 1) if (maxvote != yTest[x]) else errvote)
            erravgUn = ((erravgUn + 1) if (maxavgUn != yTest[x]) else erravgUn)
            errrandomUn = ((errrandomUn + 1) if (maxrandomUn != yTest[x]) else errrandomUn)
            errlastN = ((errlastN + 1) if (maxlastN != yTest[x]) else errlastN)
            erravgN = ((erravgN + 1) if (maxavgN != yTest[x]) else erravgN)
            errrandomN = ((errrandomN + 1) if (maxrandomN != yTest[x]) else errrandomN)
        #mistake counts the total number of errors in a certain era
        mistake = errlastUn + errvote + erravgUn + errrandomUn + errlastN + erravgN + errrandomN
        #error rate in percent committed in a certain epoch
        errlastUnPerc = (errlastUn / len(yTest)) * 100
        errvotePerc = (errvote / len(yTest)) * 100
        erravgUnPerc = (erravgUn / len(yTest)) * 100
        errrandomUnPerc = (errrandomUn / len(yTest)) * 100
        errlastNPerc = (errlastN / len(yTest)) * 100
        erravgNPerc = (erravgN / len(yTest)) * 100
        errrandomNPerc = (errrandomN / len(yTest)) * 100
        errEpoch = [errlastUnPerc, errvotePerc, erravgUnPerc, errrandomUnPerc,
                    errlastNPerc, erravgNPerc, errrandomNPerc, mistake]
        err.append(errEpoch)
    #err contains the set of error committed at each epoch
    return err


def prepareTestComplete(xTrain, yNewTrain, VIC, CC, xTest, yTest, d):
    # not implemented result
    err = []
    for i in range(0, len(VIC)):
        print("test epoch", i+1)
        errlastUn = 0
        errvote = 0
        erravgUn = 0
        errrandomUn = 0
        errlastN = 0
        erravgN = 0
        errrandomN = 0
        for x in range(0, len(xTest)):
            print("test", x)
            lastUn = []
            vote = []
            avgUn = []
            randomUn = []
            lastN = []
            avgN = []
            randomN = []
            for j in range(0, len(VIC[i])):
                #value of lastUn, vote, ... of a specific class l
                value = p.methodPredictions(xTest[x], VIC[i][j], CC[i][j], xTrain, yNewTrain[j], d)
                lastUn.append(value[0])
                vote.append(value[1])
                avgUn.append(value[2])
                randomUn.append(value[3])
                lastN.append(value[4])
                avgN.append(value[5])
                randomN.append(value[6])
            #search the highest prediction value of all classes (the highest value of lastUn, Vote, ....)
            maxlastUn = np.argmax(lastUn)
            maxvote = np.argmax(vote)
            maxavgUn = np.argmax(avgUn)
            maxrandomUn = np.argmax(randomUn)
            maxlastN = np.argmax(lastN)
            maxavgN = np.argmax(avgN)
            maxrandomN = np.argmax(randomN)
            # sum 1 if the prediction is false
            errlastUn = ((errlastUn + 1) if (maxlastUn != yTest[x]) else errlastUn)
            errvote = ((errvote + 1) if (maxvote != yTest[x]) else errvote)
            erravgUn = ((erravgUn + 1) if (maxavgUn != yTest[x]) else erravgUn)
            errrandomUn = ((errrandomUn + 1) if (maxrandomUn != yTest[x]) else errrandomUn)
            errlastN = ((errlastN + 1) if (maxlastN != yTest[x]) else errlastN)
            erravgN = ((erravgN + 1) if (maxavgN != yTest[x]) else erravgN)
            errrandomN = ((errrandomN + 1) if (maxrandomN != yTest[x]) else errrandomN)
        #error rate in percent committed in a certain epoch
        mistake = errlastUn + errvote + erravgUn + errrandomUn + errlastN + erravgN + errrandomN
        # error rate in percent committed in a certain epoch
        errlastUnPerc = (errlastUn / len(yTest)) * 100
        errvotePerc = (errvote / len(yTest)) * 100
        erravgUnPerc = (erravgUn / len(yTest)) * 100
        errrandomUnPerc = (errrandomUn / len(yTest)) * 100
        errlastNPerc = (errlastN / len(yTest)) * 100
        erravgNPerc = (erravgN / len(yTest)) * 100
        errrandomNPerc = (errrandomN / len(yTest)) * 100
        errEpoch = [errlastUnPerc, errvotePerc, erravgUnPerc, errrandomUnPerc,
                    errlastNPerc, erravgNPerc, errrandomNPerc, mistake]
        err.append(errEpoch)
    #err contains the set of error committed at each epoch
    return err


def AllTest(xTrain01, yNewTrain01, VIC01, CC01, xTrain, yNewTrain, VIC, CC, xTest, yTest, d):
    err01 = prepareTest01(xTrain01, yNewTrain01, VIC01, CC01, xTest, yTest, d)
    err = prepareTest(xTrain, yNewTrain, VIC, CC, xTest, yTest, d)
    plotGraphic(err01, err, d)


def plotGraphic(err01, err, d):
    if(d==1):
        epoch = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for j in range(1, 11, 1):
            epoch.append(j)
        errlastUn = []
        errvote = []
        erravgUn = []
        errrandomUn = []
        for i in range(0, len(err01)):
            errlastUn.append(err01[i][0])
            errvote.append(err01[i][1])
            erravgUn.append(err01[i][2])
            errrandomUn.append(err01[i][3])
        for i in range (0, len(err)):
            errlastUn.append(err[i][0])
            errvote.append(err[i][1])
            erravgUn.append(err[i][2])
            errrandomUn.append(err[i][3])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(epoch, errlastUn, 'r-')
        ax.semilogx(epoch, errvote, 'b-')
        ax.semilogx(epoch, erravgUn, 'y-')
        ax.semilogx(epoch, errrandomUn, 'g-')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(ymin=0)
        ax.set_title('d = 1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Error')
        plt.legend(['lastUn', 'vote', 'avgUn', 'randomUn'])
        plt.show()
        plt.close()
    else:
        epoch = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for j in range(1, 31, 1):
            epoch.append(j)
        errlastUn = []
        errvote = []
        erravgUn = []
        errrandomUn = []
        for i in range(0, len(err01)):
            errlastUn.append(err01[i][0])
            errvote.append(err01[i][1])
            erravgUn.append(err01[i][2])
            errrandomUn.append(err01[i][3])
        for i in range(0, len(err)):
            errlastUn.append(err[i][0])
            errvote.append(err[i][1])
            erravgUn.append(err[i][2])
            errrandomUn.append(err[i][3])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.semilogx(epoch, errlastUn, 'r-')
        ax.semilogx(epoch, errvote, 'b-')
        ax.semilogx(epoch, erravgUn, 'y-')
        ax.semilogx(epoch, errrandomUn, 'g-')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(ymin=0)
        ax.set_title('d = 2')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Error')
        plt.legend(['lastUn', 'vote', 'avgUn', 'randomUn'])
        plt.show()
        plt.close()
