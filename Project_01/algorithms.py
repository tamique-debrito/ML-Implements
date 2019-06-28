import data_opener
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import DecisionTree
import KNearestNeighbors


random.seed()

# Data formattted as list of (features, label) pairs
aawData, names = data_opener.getData("Absenteeism_at_work.csv")
random.shuffle(aawData)
print(names)

trainData = aawData[0:500]
testData = aawData[500:]


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
Some preprocessing tools
"""

# Format data for binary decision tree
# cutoffs[0] is a list of cutoffs by feature and cutoffs[1] is the cutoff for the label
def categorizeBinary(data, cutoffs):
    return [[[featureVal <= cutoff for featureVal, cutoff in zip(example[0], cutoffs[0])],example[1] <= cutoffs[1]] for example in data]
def categorizeValue(val, cutoffs, upperInclusive = False):
    cutoffs.append(float('inf'))
    if upperInclusive:
        for i in range(len(cutoffs)):
            if val <= cutoffs[i]:
                return i
    else:
        for i in range(len(cutoffs)):
            if val < cutoffs[i]:
                return i
def categorizeLabels(data, cutoffs):
    return [[example[0], categorizeValue(example[1], cutoffs)] for example in data]
def categorizeData(data, cutoffsList):
    return [[[categorizeValue(featureVal, cutoffs) for featureVal, cutoffs in zip(example[0], cutoffsList[0])], categorizeValue(example[1], cutoffsList[1])] for example in data]
aawDataCutoffs = [[None, 22, 7, 5, 2.5, 250, 35, 15, 40, 275, 90, 0.5, 2.5, 3, 0.5, 0.5, 3, 85, 175, 30], 5] #applies to data with "ID" and "Hit Target" features removed

aawDataBinary = categorizeBinary(aawData, aawDataCutoffs)

aawDataBinary, names = data_opener.removeFeatures(aawDataBinary, names, [0,1,3])

"""
Try binary decision tree
"""

#plotData(aawDataBinary, names)

classificationPercentError = lambda testData, predictor: float(len([None for example in testData if predictor.predict(example[0]) != example[1]]))/float(len(testData))

T = DecisionTree.DecisionTree()
T.train(aawDataBinary, 17)
print("Decision Tree Error: " + str(classificationPercentError(testData, T)) + "%")

getLabels = lambda data : [example[1] for example in data]

binLabel = getLabels(aawDataBinary)

"""
Try K-Nearest_Neighbors
"""
aawData, names = data_opener.removeFeatures(aawData, names, [0])
testData, names = data_opener.removeFeatures(testData, names, [0])
#aawData = categorizeLabels(aawData, [15])
#testData = categorizeLabels(testData, [15])

K = KNearestNeighbors.KNearestNeighbors()

"""
# plot error vs categorization cutoff and K
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

# plot error vs categorization cutoff

for k in [5, 10, 20, 50]:
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
plt.title("Accuracy vs cutoff for select K values")
plt.xlabel("Cutoff (Hours)", fontsize = 14)
plt.ylabel("K", fontsize = 14)

print("K-Nearest Neighbors Error: " + str(classificationPercentError(testData, K)) + "%")
