import numpy as np
import pandas as pd
import imageio
import glob
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from natsort import natsorted

def RGBConversion(image):
    R = image[:,:,0].flatten()
    G = image[:,:,1].flatten()
    B = image[:,:,2].flatten()
    bitMap = np.ceil((R+G+B) / 3)
    return bitMap

def loadImages(train_fingerprint, test_fingerprint):
    trainImages = []
    testImages = []
    count = 0
    fingerPrintListTrain=[]
    sortedListTrain= natsorted(glob.glob(train_fingerprint +'/*.bmp'))
    maxCount = len(sortedListTrain)
    for fingerprintTrain in sortedListTrain:
        im = imageio.imread(fingerprintTrain)
        if im is not None:
            row = RGBConversion(im)
            if count < maxCount:
                n = len(fingerprintTrain.split('/'))
                arr = np.asarray(row)
                arr = np.append([arr],[int(fingerprintTrain.split('/')[n-1].split('_')[0])])
                fingerPrintListTrain.append(arr.tolist())
                count += 1
        elif im is None:
            print ("Error loading: " + fingerprintTrain)
        continue
    trainImages = pd.DataFrame(fingerPrintListTrain)
    
    fingerPrintListTest=[]
    sortedListTest = natsorted(glob.glob(test_fingerprint +'/*.bmp'))
    for fingerprintTest in sortedListTest:
        im = imageio.imread(fingerprintTest)
        if im is not None:
            row = RGBConversion(im)
            arr = np.asarray(row)
            arr = np.append([arr],[int(fingerprintTest.split('/')[n-1].split('_')[0])])
            fingerPrintListTest.append(arr.tolist())
        elif im is None:
            print ("Error loading: " + fingerprintTest)
        continue
    testImages = pd.DataFrame(fingerPrintListTest)
    return trainImages, testImages


train_fingerprint, test_fingerprint = loadImages( r'/Volumes/Shared/MAC/UCDenver_CSE/MachineLearning/Assignment4/training',
                                                 r'/Volumes/Shared/MAC/UCDenver_CSE/MachineLearning/Assignment4/testB')
print('Data Loaded!!!')

xTrain = train_fingerprint.iloc[:,:-1]
xTrain = np.c_[np.ones((xTrain.shape[0])),xTrain]
yTrain = train_fingerprint.iloc[:,-1]
yTrain = yTrain.as_matrix(columns=None)

xTest = test_fingerprint.iloc[:,:-1]
xTest = np.c_[np.ones((xTest.shape[0])),xTest]
yTest = test_fingerprint.iloc[:,-1]
yTest = yTest.as_matrix(columns=None)

#Normalize the data set
xTrain = xTrain / 255
row, column = xTrain.shape[0], xTrain.shape[1]
div = sum(xTrain.sum(axis=1)) / (row * column)
xTrain = xTrain - div

xTest = xTest / 255
row, column = xTest.shape[0], xTest.shape[1]
div = sum(xTest.sum(axis=1)) / (row * column)
xTest = xTest - div

clf = OneVsRestClassifier(SVC(kernel='rbf', tol=0.03 ,C=1/0.2, gamma=0.03 ,probability=True))
clf = clf.fit(xTrain, yTrain)

accuracy = clf.score(xTest,yTest)
print('Accuracy of data: ',accuracy*100)

clf.decision_function(xTrain)
predLabel = clf.predict(xTest)
correct = np.sum(predLabel == yTest)
print("%d out of %d predictions correct" % (correct, len(predLabel)))
print("The fingerprint belongs to person %f" % (predLabel))