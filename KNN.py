# -*- coding: utf-8 -*-

# KNN算法
# 优点：精度高，对异常值不敏感，无数据输入假定
# 缺点：计算复杂，多维数据复杂度高

from numpy import *
import operator
# import matplotlib
# import matplotlib.pyplot as plt
import os
from time import time

def dataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    label = ['A','A','B','B']
    return group, label 

def classify(inX, dataSet, label, k):
    dataSetSize = dataSet.shape[0]
    temp1 = tile(inX, (dataSetSize,1 )) - dataSet
    diff = temp1 ** 2
    Distance = (diff.sum(1)) ** 0.5
    SortedDistIndex = Distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = label[SortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    SortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return SortedClassCount[0][0]

def file2Matrix(filename):
    fr = open(filename)
    fileLines = fr.readlines()
    NumLines = len(fileLines)
    returnMat = zeros((NumLines, 3))
    index = 0
    label = []
    for line in fileLines:
        line.strip()
        listLines = line.split('\t')
        returnMat[index, :] = listLines[0:3]
        label.append(int(listLines[-1]))
        index += 1
    return returnMat, label

def autoNorm(dataset):
    minVal = dataset.min(0)
    maxVal = dataset.max(0)
    m = shape(dataset)[0]
    temp1 = dataset - tile(minVal,(m,1))
    ranges = maxVal - minVal
    dataNorm = temp1/tile(ranges,(m,1))
    return dataNorm, ranges

def Testing():
    dataset, label = file2Matrix('datingTestSet2.txt')
    m = shape(dataset)[0]
    dataNorm, ranges = autoNorm(dataset)
    TestVec = int(m*0.10)
    error = 0
    for i in range(TestVec):
        ans = classify(dataNorm[i,:], dataNorm[TestVec:m,:], label[TestVec:m], 5)
        print('the classifer result is {}, the real answer is {}'.format(ans, label[i]))
        if (ans != label[i]): error += 1
    accu = error/float(TestVec)
    print('the total error is {}, the accuracy is {}'.format(error, 1-accu))

def classPerson():
    iceCream = float(input('liters of ice cream consumed per year?'))
    videoGame = float(input('percentage of time spend on video game?'))
    flight = float(input('frequent flier miles eared per year?'))
    resultList = ['not at all', 'in small doses', 'in large doses']
    dataset, label = file2Matrix('datingTestSet.txt')
    dataNorm, ranges = autoNorm(dataset)
    inX = array([flight, videoGame, iceCream])
    ans = classify(inX, dataNorm, label, 5)
    print('the person is {} for you'.format(resultList[ans-1]))

def img2vec(filename):
    returnVec = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVec[0,i*32+j] = int(line[j])
    return returnVec

def handwritingTest():
    start = time()
    hwLabel = []
    fileList = os.listdir('/digits/trainingDigits')
    mTrain = len(fileList)
    TrainVec = zeros((mTrain, 1024))

    for i in range(mTrain):
        filename = fileList[i]
        temp1 = filename.split('.')[0]
        label = temp1.split('_')[0]
        hwLabel.append(label)
        TrainVec[i,:] = img2vec('/digits/trainingDigits/{}'.format(filename))

    testList = os.listdir('/digits/testDigits')
    mTest = len(testList)
    # testLabel = []
    # TestVec = zeros((mTest, 1024))
    error = 0.0
    for j in range(mTest):
        testFile = testList[j]
        temp2 = testFile.split('.')[0]
        testLabel = temp2.split('_')[0]
        # testLabel.append(tlabel)
        TestVec = img2vec('/digits/testDigits/{}'.format(testFile))
        ans = classify(TestVec, TrainVec, hwLabel, 5)
        print('the classifer come out with {}, the real answer is {}'.format(ans, testLabel))
        if (int(ans) != int(testLabel)): error += 1
    errRate = error/float(mTest)
    end = time()
    print('the total error is {}.'.format(error))
    print('the test accuracy is {}%.'.format((1-errRate)*100))
    print('the operating time is {}s.'.format(end-start))
