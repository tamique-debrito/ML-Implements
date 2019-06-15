import DecisionTree
D = DecisionTree()

n = D.trainAux([[[False, True], True], [[True, True], False]],[0])
drawTree(n, 0)


n = D.trainAux([[[False, True], True], [[True, True], False],
                [[True, False], True], [[False, False], False]],[0,1])
drawTree(n, 0)


n = D.trainAux([[[False, True], True], [[True, True], False],
                [[True, False], True], [[False, False], False]],[0])
drawTree(n, 0)
