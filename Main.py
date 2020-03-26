from utils import zalandoReader as rz
from utils import helper as h
import training as t
import test
from configs import ZALANDO_DIR
import matplotlib.pyplot as plt


def main():
    # zalando dataset
    # train
    Xtrain, Ytrain = rz.load_zalando_train(path=ZALANDO_DIR)
    labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
    # test
    Xtest, Ytest = rz.load_zalando_t10k(path=ZALANDO_DIR)
    plt.imsave('zalandoDataset-sprite-train.png', h.get_sprite_image(Xtrain), cmap='gray')
    plt.close()
    plt.imsave('zalandoDataset-sprite-test.png', h.get_sprite_image(Xtest), cmap='gray')
    plt.close()
    xReduced01 = Xtrain[0:6000].copy()
    yReduced01 = Ytrain[0:6000].copy()
    VIC01, CC01, yNC01, VIC, CC, yNC = t.trainingD(Xtrain, Ytrain,
                                                   xReduced01, yReduced01, labels, 1)
    test.allTest(xReduced01, yNC01, VIC01, CC01,
                 Xtrain, yNC, VIC, CC,
                 Xtest, Ytest, 1)

    VIC01, CC01, yNC01, VIC, CC, yNC = t.trainingD(Xtrain, Ytrain,
                                                   xReduced01, yReduced01, labels, 2)
    test.allTest(xReduced01, yNC01, VIC01, CC01,
                 Xtrain, yNC, VIC, CC,
                 Xtest, Ytest, 2)


if __name__ == '__main__':
    main()

