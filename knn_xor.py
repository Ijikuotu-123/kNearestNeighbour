# knn can't be used to solve grid of alternating points. traininig accuracy = 0

from Knn import KNN
from util import get_xor
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, Y = get_xor()

    plt.scatter(X[:,0], X[:,1], s=100, c =Y, alpha = 0.5)
    plt.show()

    model = KNN(3)
    model.fit(X, Y)
    print ("Accuracy:", model.score(X,Y))