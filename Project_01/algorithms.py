# Created by Tamique de Brito.
# Updated 2 July, 2019

import copy
import data_opener
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import DecisionTree
import MultiDecisionTree
import KNearestNeighbors
import Perceptron
import AveragedPerceptron


random.seed()

# Data formattted as list of (features, label) pairs
mainData, featureNames = data_opener.getData("Absenteeism_at_work.csv")
random.shuffle(mainData)

# Data labels
#print(names)

trainData = mainData[0:500]
testData = mainData[500:]


# Plots data in histogram form, where data is given as list of (featurelist, label) pairs
#   and names is a list of feature names corresponding to entries of each featurelist
def plotData(data, names):
    # Data formatted as list of featureValues_i, where featureValues_i
    #   is a list of all the values of feature i in the data set
    valuesByFeature, labelValues = data_opener.transpose(data)

    numFeatures = len(valuesByFeature)
    
    dim = int(math.ceil(math.sqrt(numFeatures)))
    
    # features plots
    fig1 = plt.figure()
    axes = []
    
    featuresRemaining = numFeatures
    for i in range(dim):
        for j in range(dim):
            if featuresRemaining == 0:
                break
            axes.append(fig1.add_subplot(dim,dim,i*dim + j+1))
            featuresRemaining -= 1
    for ax, dat, name in zip(axes, valuesByFeature, names):
        ax.hist(dat)
        ax.title.set_text(name)
    
    fig1.set_size_inches(18.5, 10.5)
    fig1.tight_layout()
    fig1.show()

"""
# Plot label histogram
valuesByFeature, labelValues = data_opener.transpose(aawData)
fig2 = plt.figure()
fig2.add_subplot(1,2,1)
fig2.set_size_inches(10, 5)
plt.hist(labelValues)
plt.xlabel('Hours')
plt.ylabel('Number')
plt.title('Time Absent (Lin Scale)')
fig2.add_subplot(1,2,2)
plt.hist(labelValues, log=True)
plt.xlabel('Hours')
plt.ylabel('Number')
plt.title('Time Absent (Log Scale)')
"""


"""
Error calculator
"""

classificationPercentError = lambda testData, predictor: float(len([None for example in testData if predictor.predict(example[0]) != example[1]]))/float(len(testData))





"""
Some preprocessing tools
"""

# Format data for binary decision tree
# cutoffs[0] is a list of cutoffs by feature and cutoffs[1] is the cutoff for the label
def categorizeBinary(data, cutoffs):
    return [[[featureVal <= cutoff for featureVal, cutoff in zip(example[0], cutoffs[0])],example[1] <= cutoffs[1]] for example in data]
def categorizeValue(val, cutoffs, upperInclusive = False):
    if cutoffs == None or cutoffs[0] == None:
        return 0
    if upperInclusive:
        for i in range(len(cutoffs)):
            if val <= cutoffs[i]:
                return i
        return len(cutoffs)
    else:
        for i in range(len(cutoffs)):
            if val < cutoffs[i]:
                return i
        return len(cutoffs)
def categorizeLabels(data, cutoffs):
    return [[example[0], categorizeValue(example[1], cutoffs)] for example in data]
def categorizeData(data, cutoffsList):
    return [[[categorizeValue(featureVal, cutoffs) for featureVal, cutoffs in zip(example[0], cutoffsList[0])], categorizeValue(example[1], cutoffsList[1])] for example in data]
aawDataBinaryCutoffs = [[None, 22, 7, 5, 2.5, 250, 35, 15, 40, 275, 90, 0.5, 2.5, 3, 0.5, 0.5, 3, 85, 175, 30], 5] #applies to data with "ID" and "Hit Target" features removed


aawDataBinary = categorizeBinary(trainData, aawDataBinaryCutoffs)
aawDataBinary, names = data_opener.removeFeatures(aawDataBinary, featureNames, [0,1,3])

testDataBinary = categorizeBinary(testData, aawDataBinaryCutoffs)
testDataBinary, _ = data_opener.removeFeatures(testDataBinary, featureNames, [0,1,3])


















"""
Try binary decision tree
"""

#plotData(aawDataBinary, names)


T = DecisionTree.DecisionTree()
T.train(aawDataBinary, 17)
print("Binary Decision Tree Error: " + str(classificationPercentError(testDataBinary, T)) + "%")

# Plot: Error of binary decision tree v.s. cutoff for label classification
"""
hoursCutoffs = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20, 30, 40, 50]
accuracies = []
for j in range(15):
    random.shuffle(mainData)
    aawTrainData = mainData[0:500]
    aawTestData = mainData[500:]
    accuracies.append([])
    for i in hoursCutoffs:
        aawDataCutoffs[1] = i
        aawTrainDataBinary = categorizeBinary(aawTrainData, aawDataCutoffs)
        aawTrainDataBinary, _ = data_opener.removeFeatures(aawTrainDataBinary, names, [0,1,3])
        
        aawTestDataBinary = categorizeBinary(aawTestData, aawDataCutoffs)
        aawTestDataBinary, _ = data_opener.removeFeatures(aawTestDataBinary, names, [0,1,3])
        T.train(aawTrainDataBinary, 17)
        accuracies[j].append(classificationPercentError(aawTestDataBinary, T))
plt.plot(hoursCutoffs, np.average(accuracies, axis=0))
plt.title("Error vs cutoff", fontsize = 18)
plt.xlabel("Cutoff (Hours)", fontsize = 14)
plt.ylabel("Error", fontsize = 14)
"""

getLabels = lambda data : [example[1] for example in data]
binLabel = getLabels(aawDataBinary)





# Plot True/False distribution of labels after splitting by feature True/False for each feature

"""
# Get proportion of examples labelled true out of the examples in data
#       where feature f has truth value T
getP = lambda data, f, T: float(len([e for e in aawDataBinary if e[0][f] == T and e[1]]))/float(len([e for e in aawDataBinary if e[0][f] == T]))
fCount = lambda f: [getP(aawDataBinary, f, True), 1-getP(aawDataBinary, f, True), getP(aawDataBinary, f, False), 1 - getP(aawDataBinary, f, False)]

plt.bar([0,1, 3,4], fCount(9))
plt.title("Distribution of label True/False for [feature] = True/False")
plt.bar([0,1, 3,4], fCount(3))
plt.title("Distribution of label True/False for [feature] = True/False")

figBar = plt.figure()
axesBar = []

for i in range(16):
    axesBar.append(figBar.add_subplot(4,4,i + 1))
for i in range(16):
    axesBar[i].bar([0,1, 3,4], fCount(i))
    axesBar[i].title.set_text(names[i])

figBar.set_size_inches(18, 10)
figBar.tight_layout()
figBar.show()
"""













"""
Try Multi-Decision Tree
"""

"""
removedFeatures = [0, 1, 3]

labelCutoffs = [1,2,3,4,5,6,7,10]

aawDataCutoffs = [[[1], [22], [1,2,3,4,5,6,7,8,9,10,11], [1,2,3,4,5,6,7], 
                   [1,2,3,4], [100,150,200,300], [10,15,20,40], [5,10,20], [20,30,40,50],
                   [250,300,350], [90,95], [0.5], [1,2,3], [0,1,2,3,4], 
                   [0.5], [0.5], [0,1,2,4,6], [70,80,90,100], [170,180,190],[22,29,35]],
                    labelCutoffs]
aawDataCutoffs_f = [[copy.deepcopy(aawDataCutoffs[0][i]) for i in range(len(aawDataCutoffs[0])) if not i in removedFeatures], copy.deepcopy(aawDataCutoffs[1])]

aawTrainData = categorizeData(trainData, aawDataCutoffs)
aawTrainData, names = data_opener.removeFeatures(aawTrainData, names, removedFeatures)
aawTestData = categorizeData(testData, aawDataCutoffs)
aawTestData, _ = data_opener.removeFeatures(aawTestData, names, removedFeatures)



MT = MultiDecisionTree.MultiDecisionTree()
MT.train(aawTestData, len(aawDataCutoffs_f[0]), map(lambda x: len(x) + 1, aawDataCutoffs_f[0]), len(aawDataCutoffs[1]) + 1, preformatted=True)
print("Multi Decision Tree Error: " + str(classificationPercentError(aawTestData, MT)) + "%")

labelCount = lambda data, labels: [len([False for d in data if d[1] == l]) for l in labels]
predict = lambda data, predictor: [[d[0], predictor.predict(d[0])] for d in data]


fig, ax = plt.subplots()
rects1 = ax.bar([i + 0.3 for i in labelCutoffs], labelCount(aawTestData, labelCutoffs), width=0.2, label='actual')
rects2 = ax.bar(labelCutoffs, labelCount(predict(aawTestData, MT), labelCutoffs), width=0.2, label='predicted')
ax.set_title("Multi Decision Tree: \nactual distribution of test data labels \nvs distribution of predicted data labels")

ax.legend()
fig.tight_layout()
plt.show()
"""
















"""
Try K-Nearest-Neighbors
"""

K = KNearestNeighbors.KNearestNeighbors()

K.setData(aawDataBinary, 2, 2)

print("K-Nearest Neighbors Error: " + str(classificationPercentError(testDataBinary, K)) + "%")

# Plot: error vs categorization cutoff and K
"""
cutoffSeq = []
kSeq = []
accuracySeq = []
for i in [1,2,3,4,5,6,8,11,14]:
    for j in [5, 10, 15, 20, 25, 30]:
        trainData_i = categorizeLabels(aawData, [i])
        testData_i = categorizeLabels(testData, [i])
        K.setData(trainData_i, 2, j)
        cutoffSeq.append(i)
        kSeq.append(j)
        accuracySeq.append(20/(classificationPercentError(testData_i, K)+0.1))
fig, ax = plt.subplots()
ax.scatter(cutoffSeq, kSeq, s=accuracySeq)
ax.set_xlabel("Cutoff", fontsize = 14)
ax.set_ylabel("K", fontsize = 14)
ax.set_title("Accuracy vs cutoff and K (larger dots = more accurate)", fontsize = 18)
plt.show()
"""


# Plot: error vs categorization cutoff
"""
for k in [1, 2, 5, 10, 20, 50]:
    cutoffSeq = []
    accuracySeq = []
    for i in range(15):
        trainData_i = categorizeLabels(aawData, [i])
        testData_i = categorizeLabels(testData, [i])
        K.setData(trainData_i, 2, k)
        cutoffSeq.append(i)
        accuracySeq.append(classificationPercentError(testData_i, K))
    plt.plot(cutoffSeq, accuracySeq, label = "K = " + str(k))
plt.legend()
plt.title("Error vs cutoff for select K values")
plt.xlabel("Cutoff (Hours)", fontsize = 14)
plt.ylabel("Error", fontsize = 14)
"""


# Plot: error v.s. number of cutoffs used
"""
cutoffSeq = []
accuracySeq = []
for k in [1, 2, 5, 10, 20, 50]:
    cutoffSeq = []
    accuracySeq = []
    for c in range(5):
        trainData_i = categorizeLabels(aawData, [i+1 for i in range(c)])
        testData_i = categorizeLabels(testData, [i+1 for i in range(c)])
        K.setData(trainData_i, c+2, k)
        cutoffSeq.append(c)
        accuracySeq.append(classificationPercentError(testData_i, K))
    plt.plot(cutoffSeq, accuracySeq, label = "K = " + str(k))
plt.legend()
plt.title("Error vs num. cutoffs")
plt.xlabel("Cutoffs (#)", fontsize = 14)
plt.ylabel("Error", fontsize = 14)
"""






















"""
Try Perceptron
"""

labelCutoff = 3
aawTrainDataBinLabels = [[np.array([f for f in e[0]]), 1 if e[1] > labelCutoff else -1] for e in trainData]
aawTestDataBinLabels = [[np.array([f for f in e[0]]), 1 if e[1] > labelCutoff else -1] for e in testData]

removedFeatures = [0, 1, 3, 6, 7, 10, 13, 19, 20]


aawTrainDataBinLabels, _ = data_opener.removeFeatures(aawTrainDataBinLabels, [], removedFeatures)
aawTestDataBinLabels, _ = data_opener.removeFeatures(aawTestDataBinLabels, [], removedFeatures)

aawTrainDataBinLabels = map(lambda e: [np.array(e[0]), e[1]], aawTrainDataBinLabels)
aawTestDataBinLabels = map(lambda e: [np.array(e[0]), e[1]], aawTestDataBinLabels)


P = Perceptron.Perceptron()
P.train(np.array(aawTrainDataBinLabels), 50, len(aawTrainDataBinLabels[0][0]))

print("Perceptron Error: " + str(classificationPercentError(aawTestDataBinLabels, P)) + "%")


AP = AveragedPerceptron.AveragedPerceptron()
AP.train(np.array(aawTrainDataBinLabels), 50, len(aawTrainDataBinLabels[0][0]))

print("Averaged Perceptron Error: " + str(classificationPercentError(aawTestDataBinLabels, AP)) + "%")
