import DecisionTree

import copy

D = DecisionTree.DecisionTree()

D.train([[[False, True], True], [[True, True], False]],1)
DecisionTree.drawTree(D.root, 0)
print([True, False], D.predict([True, False]))

D.train([[[False, True], True], [[True, True], False],
                [[True, False], True], [[False, False], False]],2)
DecisionTree.drawTree(D.root, 0)
print([False, True], D.predict([False, True]))

D.train([[[False, True], True], [[True, True], False],
                [[True, False], True], [[False, False], False]], 1)
DecisionTree.drawTree(D.root, 0)
print([False, True], D.predict([False, True]))




# Some code for generating more tests
def genA(List, n):
    if n == 1:
        return [l + [True] for l in List] + [l + [False] for l in List]
    return genA([l + [True] for l in List] + [l + [False] for l in List], n-1)
def parity(blist):
    p = True
    for b in blist:
        p = p != b
    return p
def gen(n):
    L = genA([[]],n)
    return [[l, parity(l)] for l in L]

n = D.train(gen(4), 4)
DecisionTree.drawTree(D.root, 0)
print([False]*4, D.predict([False]*4))
print([True]*4, D.predict([True]*4))
print([False, True]*2, D.predict([False, True]*2))
