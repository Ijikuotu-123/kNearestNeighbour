# To install sortedcontainers do: sudo pip install sortedcontainers

import numpy as np
from sortedcontainers import SortedList
# you can't use sortedDict because the key is distance and if 2 close points are
# the same distance away, one will be overwritten
from util import get_data
from datetime import datetime

class kNN(object):
    def __init__ (self,k):
        self.k = k

    def fit (self, X,Y):   # train. this saves x and y
        self.X = X
        self.Y = Y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList(load=self.k)  # load tells us the size of the sorted list
            for j , xt in enumerate(self.X):
                diff = x -xt
                d=diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((d,self.Y[j]))
                else:
                    if d < sl[-1][0]:   # [-1] is the last item in the list and [0] is the first item in the choosen list
                        del sl[-1]
                        sl.add((d,self.Y[j]))

            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self,X,Y): 
        P = self.predict(X)
        return np.mean(P ==Y)

if __name__ == " __main__":    
    X.Y = get_data(2000)
    Ntrain = 1000
    Xtrain, Ytrain = X[:Ntrain],Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:],Y[Ntrain:]
    for k in (1,2,3,4,5):
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain,Ytrain)
        print ('training time:', datetime.now() -t0)

        t0 = datetime.now()
        print('training accuracy:', knn.score(Xtrain, Ytrain))
        print('time to compute train accuracy:', (datetime.now() -t0), 'Train size:', len(Ytrain))

        t0 = datetime.now()
        print('testing accuracy:', knn.score(Xtest, Ytest))
        print('time to compute test accuracy:', (datetime.now() -t0), 'Test size:', len(Ytest))

""" generally, test accuracy will be less than train accuracy"""







    
