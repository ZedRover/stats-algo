import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib # A must 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection as skms
from sklearn import metrics
from sklearn import neighbors
from sklearn import naive_bayes
from time import sleep

import memory_profiler, sys

@memory_profiler.profile(precision=4) #decorator
def knn_memtest(train, train_tgt, test):
    knn   = neighbors.KNeighborsClassifier(n_neighbors=3)
    fit   = knn.fit(train, train_tgt)
    preds = fit.predict(test)

if __name__ == "__main__":
    iris = datasets.load_iris()
    tts = skms.train_test_split(iris.data,
                                iris.target,
                               test_size=.25)
    (iris_train_ftrs, iris_test_ftrs,
     iris_train_tgt,  iris_test_tgt) = tts
    tup = (iris_train_ftrs, iris_train_tgt, iris_test_ftrs)
    knn_memtest(*tup)
