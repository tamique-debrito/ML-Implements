import csv

#returns a list of items in format (tupleOfFeatures, label)
def getData(filename, transpose = False):
    csvFile = open(filename)
    L = csv.reader(csvFile, delimiter = ';')
    next(L)

    data = []
    
    for row in L:
        data.append([[float(item) for item in row[:-1]], int(row[-1])])

    
    
    if transpose:
        numDataPoints = len(data); numFeatures = len(data[0][0])
        return [[data[i][0][j] for i in range(numDataPoints)] for j in range(numFeatures)]
    else:
        return data

def transpose(data):
    numDataPoints = len(data); numFeatures = len(data[0][0])
    return [[data[i][0][j] for i in range(numDataPoints)] for j in range(numFeatures)]

#truncates lists in data to length at most n
def truncate(data, n):
    return [[l[i] for i in range(min(n, len(l)))] for l in data]