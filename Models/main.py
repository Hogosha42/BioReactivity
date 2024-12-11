import numpy as np
from functools import lru_cache
import rdkit

def getColumn(matrix, columnNum:int):
    return [x[columnNum] for x in matrix]

def pca(data:list[tuple]):
    # returns the principal components and their respective eigenvectors
    meanrow = np.array(map(np.mean, [getColumn(data, i) for i in range(len(data))]))
    B = np.array(data) - np.outer(np.array([1 for _ in range(len(data[0]))]), meanrow) #mean centered data
    U,S,V = np.linalg.svd(B) # Singular value decomposition
    return np.dot(U,S), np.vectorize(lambda x:x**2)(S)

def mapi(fun, ls):
	return list(map(fun, range(len(ls)),ls))

def scaling(data:list[tuple]):
    @lru_cache(maxsize=50000)
    def scalefunc(x:float, column:list):
        return (x-min(column))/(max(column)-min(column))
    return mapi(lambda i,x:list(map(lambda y:scalefunc(y, getColumn(i)), x)), data)

def balancedAccuracy(data:list, prediction:list):
    truepos = sum(yhat==1==y for y,yhat in zip(data, prediction))
    trueneg = sum(yhat==0==y for y,yhat in zip(data, prediction))
    return ((truepos/sum(y==1 for y in prediction))+(trueneg/sum(y==0 for y in prediction)))*0.5


if __name__ == "__main__":
    print()