import numpy as np
from functools import reduce, lru_cache

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

traindata = GetFileData("Datasets/train.csv", True, [lambda x: x[1:-1], lambda x: 0 if x[1]=="0" else 1])
testdata = GetFileData("Datasets/test.csv", True, [int, lambda x: x[1:-1]])

def getColumn(matrix, columnNum:int):
    return [x[columnNum] for x in matrix]

def autoscale(data):
    
    def scale(mu, sd):
        return np.vectorize(lambda v:(v-mu)/sd if sd!=0 else 0)
    
    return np.array([scale(np.mean(x), np.std(x))(x) for x in np.array(data).T]).T

def PCA(data):
    scaleddata = autoscale(data)
    covmat = np.cov(scaleddata)
    U,S,V = np.linalg.svd(covmat)

    return np.dot(scaleddata, V), np.vectorize(lambda x:x**2)(S)

def mapi(fun, ls):
	return list(map(fun, range(len(ls)),ls))

def scaling(data:list[tuple]):
    def scalefunc(x:float, column:list):
        return (x-min(column))/(max(column)-min(column))
    return mapi(lambda i,x:list(map(lambda y:scalefunc(y, getColumn(i)), x)), data)

def balancedAccuracy(data:list, prediction:list):
    truepos = sum(yhat==1==y for y,yhat in zip(data, prediction))
    trueneg = sum(yhat==0==y for y,yhat in zip(data, prediction))
    return ((truepos/sum(y==1 for y in prediction))+(trueneg/sum(y==0 for y in prediction)))*0.5


if __name__ == "__main__":
    pass