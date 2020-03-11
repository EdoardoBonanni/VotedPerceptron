import matplotlib.pyplot as plt
import numpy as np
from utils import zalandoReader as rs
from utils import helper as h
from configs import ZALANDO_DIR
import math

def main():
    #train
    Xtrain, Ytrain = rs.load_zalando_train(path=ZALANDO_DIR)
    labelsTrain = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    YstrTrain = np.array([labelsTrain[j] for j in Ytrain])
    """
    np.savetxt('Xtrain.tsv', Xtrain, fmt='%.6e', delimiter='\t')
    np.savetxt('Ytrain.tsv', Ytrain, fmt='%.6e', delimiter='\t')
    np.savetxt('YstrTrain.tsv', YstrTrain, fmt='%s')
    plt.imsave('zalando-mnist-sprite-train.png', h.get_sprite_image(Xtrain), cmap='gray')
    plt.close()
    """

    #test
    Xtest, Ytest = rs.load_zalando_t10k(path=ZALANDO_DIR)
    labelsTest = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                  'ankle_boots']
    YstrTest = np.array([labelsTest[j] for j in Ytest])
    """
    np.savetxt('Xtest.tsv', Xtest, fmt='%.6e', delimiter='\t')
    np.savetxt('Ytest.tsv', Ytest, fmt='%.6e', delimiter='\t')
    np.savetxt('YstrTest.tsv', YstrTest, fmt='%s')
    plt.imsave('zalando-mnist-sprite-test.png', h.get_sprite_image(Xtest), cmap='gray')
    plt.close()
    """
    """
    print("X:", X)
    print("Y:", Y)
    print("Y_str:", Y_str)
    """
    """
    lenXrows = len(Xtrain)
    lenXcolumns = len(Xtrain[0])
    lenY = len(Ytrain)
    lenYstr = len(YstrTrain)
    print(lenXrows, lenXcolumns, lenY, lenYstr)
    print(Xtrain, Ytrain, labelsTrain, YstrTrain)
    print("Reduced X")
    e = math.floor(len(Xtrain)/10)
    XreducedRows = Xtrain[0:e]
    lenXredRows = len(XreducedRows)
    print(XreducedRows)
    print(lenXredRows, len(XreducedRows[0]))
    """

    

if __name__ == '__main__':
    main()