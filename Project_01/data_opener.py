import csv

# returns a pair (data, names) where
#   data is a list of items in format (tupleOfFeatures, label) and
#   names is a list of feature names
def getData(filename, transpose = False):
    csvFile = open(filename)
    L = csv.reader(csvFile, delimiter = ';')
    names = next(L)

    data = []
    
    for row in L:
        data.append([[float(item) for item in row[:-1]], int(row[-1])])

    
    
    if transpose:
        numDataPoints = len(data); numFeatures = len(data[0][0])
        return [[data[i][0][j] for i in range(numDataPoints)] for j in range(numFeatures)], names
    else:
        return data, names

# returns a pair (newData, labels) where newData is in format of a list of featureValues_i,
#   where featureValues_i is a list of all the values of feature i in the data set
#   and labels is a list of the label values
def transpose(data):
    numDataPoints = len(data); numFeatures = len(data[0][0])
    return [[data[i][0][j] for i in range(numDataPoints)] for j in range(numFeatures)], [example[1] for example in data]

#truncates lists in data to length at most n
def truncate(data, n):
    return [[l[i] for i in range(min(n, len(l)))] for l in data]

#removes features at each of the indices in "indices"
def removeFeatures(data, names, indices, transposed=False):
    if transposed:
        return [data[i] for i in range(len(data)) if i not in indices], [names[i] for i in range(len(names)) if i not in indices]
    else:
        return [[[example[0][i] for i in range(len(example[0])) if i not in indices], example[1]] for example in data], [names[i] for i in range(len(names)) if i not in indices]
