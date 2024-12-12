import numpy as np
from functools import lru_cache
import rdkit

def GetFileData(path:str, ExcludeFirstRow:bool=False, template:list=None) -> list[tuple]:
    """GetFileData(Path, *[template=list[callables], ExcludeFirstRow:bool]) -> Data:list[tuple]
    
    returns the data from a data file.
    template will be a list of callables that match the data type in each column. 
    e.g. template=[str, str, int, float, str]"""
    match path.split(".")[-1]:
        case "csv":
            with open(path) as f:
                data = [tuple(line[:-1].split(",")) for line in f.readlines()]
        case "txt":
            with open(path) as f:
                data = [tuple(line.split()) for line in f.readlines()]
        case _:
            return None
    Final = data[1:] if ExcludeFirstRow else data
    if isinstance(template, list):
        return list(map(lambda item:tuple(map(lambda x,func: func(x) ,item, template)), Final)) if bool(reduce(lambda a,b: a and b ,list(map(lambda x:len(template)==len(x) ,Final)))) else ValueError("Template and rows are not of the same length")
    else: 
        return Final

testdata = GetFileData("Datasets/test.csv", True, [int, str])


def exportCSV(data):
    return 

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


def Testmodel(data:list[tuple], model):
    return 


if __name__ == "__main__":
    print()