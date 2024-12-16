import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors, MolFromSmiles
from sklearn.decomposition import PCA

from main import GetFileData, autoscale, testdata, traindata


#smiles = [smile for smile,target in traindata]
#molecules = [(smile,MolFromSmiles(smile)) for smile in smiles]
#data = pd.DataFrame([[smile]+[d[1](m) for d in Descriptors._descList] for smile,m in molecules])

descdata = [list(map(float, row[2:])) for row in GetFileData("Datasets/descriptors.csv", True)]



if __name__ == "__main__":
    pca = PCA(n_components=.8)
    transformedData = pca.fit_transform(autoscale(descdata))
    print(transformedData)