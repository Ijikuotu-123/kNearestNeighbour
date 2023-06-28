import numpy as np
import pandas as pd


def get_data(limit=None):
    df = pd.read_csv('../large_files/train,csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data [:, 1:]/255.0   # data is from 0 ..... 255. divide by 255 t0 make data to be from 0 -1
    Y = data [:, 0]
    if limit is not None:   # setting limit so that the algorithm don't take too long. for too long data
        X, Y = X[:limit], Y[:limit]
    return X, Y
   
def get_donut():  # The  dotnut problem is when we have a class inside another class
    N = 200
    R_inner = 5
    R_outer = 10

    # distance from origin is radius  + random normal
    # angle theta is uniformlly distributed between (0, 2pi)
    R1 = np.random.random(N/2) + R_inner
    theta = 2*np.pi*np.random.random(N/2)      # polar coordinate
    X_inner = np.concatenate([[R1 *np.cos(theta)], [R1 * np.sin(theta)]]).T     # cartisean coordinate

    R2 = np.random.random(N/2) + R_outer
    theta = 2*np.pi*np.random.random(N/2)
    X_outer = np.concatenate([[R2 *np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0]*(N/2) + [1]*(N/2))
    return X, Y

def get_xor():        # modified XOR
    W = np.zeros((200, 2))
    W[:50,:] = np.random.random((50,2)) / 2 + 0.5  #(0.5 - 1, 0.5 - 1)
    W[50:100,:] = np.random.random((50,2)) / 2    #(0 - 0.5,0 - 0.5)
    W[100:150,:] = (np.random.random((50,2)) / 2) + np.array([[0,0.5]]) # (0 - 0.5,0.5 - 1)
    W[150:,:] =  (np.random.random((50,2)) / 2 ) + np.array([[0.5,0]])  # (0.5 - 1,0 - 0.5)
    Q = np.array([0]*100 + [1]*100)
    return W, Q

    
    
""" kNN will fail if the data points are alternating points e.g the XOR """





