import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import types

def copy_func(f, name=None):
    """http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    return types.FunctionType(f.func_code, f.func_globals, name or f.func_name,
        f.func_defaults, f.func_closure)


class LinearRegression:
    def __init__(self, data, dimension, learningRate):
        self.data = data
        self.learningRate = learningRate
        self.theta = [0]*dimension
        self.dim = dimension
        assert len(data[0][0]) == dimension, "dimension of data does not match given dimension"

    def train(self, maxIter, maxDiv, goal):
        lastLoss = self.loss()
        for i in range(maxIter):
            currentLoss = self.trainIteration()
            delta = currentLoss - lastLoss
            if delta > maxDiv or currentLoss < goal:
                break
    
    def trainIteration(self):
        for i in range(self.dim):
            self.theta[i] -= self.grad(i) * self.learningRate
        return self.loss()
    
    def predict(self, point):
        assert len(point) == self.dim, "point to predict has mis-matched dimension"
        return float(sum([self.theta[i] * point[i] for i in range(self.dim)]))
    
    def loss(self):
        return sum([(self.predict(d[0]) - d[1])**2 for d in self.data])/float(2*len(self.data))

    def grad(self, j):
        return sum([(self.predict(d[0]) - d[1])*d[0][j] for d in self.data])/float(len(self.data))




def graphLine(l, a, b):
    x = np.linspace(a, b, 50)
    y = np.array([l(t) for t in x])
    plt.plot(x,y,'-o',label='regression')

def graphData(data):
    x = np.array([t[0][0] for t in data])
    y = np.array([t[1] for t in data])
    plt.plot(x,y,'ro',label='data')

def graphLineAndData(data, l):
    a = min([d[0][0] for d in data])
    b = max([d[0][0] for d in data])
    x = [a,b]
    y = np.array([l(t) for t in x])
    plt.plot(x,y,'-o',label='regression')
    x = np.array([t[0][0] for t in data])
    y = np.array([t[1] for t in data])
    plt.plot(x,y,'ro',label='data')

def graphLineListAndData(data, lineList):
    fig, ax = plt.subplots()
    N = float(len(lineList))
    a = min([d[0][0] for d in data])
    b = max([d[0][0] for d in data])
    x = np.array([t[0][0] for t in data])
    y = np.array([t[1] for t in data])
    ax.plot(x,y,linestyle="None",marker='o',color=(0.1,0.1,0.9),label='data')
    i=0.0
    for l in lineList:
        ax.plot([a,b],[l(a), l(b)],linestyle='-',marker='X',color=(i,0,0,i),label=i)
        i += 1.0/N




data = [random.gauss(0,1) for i in range(20)]
m = random.random()
b = random.random()
data = [[[x, 1], m*x + b + random.gauss(0,0.1)] for x in data]

L = LinearRegression(data, 2, 0.1)


graphData(data)


numSteps = 5
trainIter = 5

for alpha in [0.01,0.05,0.1,0.2,0.5,1,5]:
    L.theta = [0,0]
    L.learningRate = alpha
    lineList = []
    
    for i in range(numSteps):
        L.train(trainIter, 1, 0)
        lineList.append(lambda t, m=L.theta[0], b=L.theta[1]: m * t + b)
    
    graphLineListAndData(data, lineList)
