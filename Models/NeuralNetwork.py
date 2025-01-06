import numpy as np; import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, InputLayer
from main import GetFileData, autoscale, traindata, scaling
import math


def projectPCA(pca, newdata):
    return pca.transform(newdata)

def PCAfit(xdata, explainedvariance=.8):
    pca = PCA(explainedvariance)
    pca.fit(xdata)
    return pca

class BalancedSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)


# --- reading x,y data from descriptors and traindata --- #
smiles, target = tuple(map(np.array, zip(*traindata))) 
descdata = pd.read_csv("Datasets/descriptors.csv").drop(columns=["0","-1"])
targetY = pd.DataFrame(target)

# --- splitting before any scaling --- #
xTrain, xTest, yTrain, yTest = train_test_split(descdata, targetY, random_state=21, test_size=.2)

# --- scale and perform principal component analysis --- #
xTrain = scaling(xTrain)
xTest = scaling(xTest)
pca = PCAfit(xTrain)
xTrainPCA = projectPCA(pca, xTrain)
xTestPCA = projectPCA(pca, xTest)
print(xTrainPCA.shape)
# --- model setup --- #

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(shape=(26,)),
    tf.keras.layers.Dense(100, activation="leaky_relu"),  # hidden layers have leak relu function to not lose information while training
    tf.keras.layers.Dense(60, activation="leaky_relu"),  
    tf.keras.layers.Dense(10, activation="leaky_relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[BalancedSparseCategoricalAccuracy])

model.fit(xTrainPCA, yTrain, epochs=200, batch_size=32, )

y = model.predict(xTestPCA)
yhh = list(map(float, model.predict(xTestPCA)))
yhat = [0 if v<0.5 else 1 for v in y]



if __name__ == "__main__":
    print(yhh)
    print(balanced_accuracy_score(yTest, yhat))