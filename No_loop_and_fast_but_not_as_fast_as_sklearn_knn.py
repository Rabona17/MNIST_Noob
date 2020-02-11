import scipy as sp
import numpy as np
import pandas as pd
def euclidean_distance(vector, point):
    return np.linalg.norm((vector - point), axis=1)

def txt_to_array(fp, dtype=int):
    f = open(fp, 'r')
    x = np.array(f.readlines())
    f.close()
    arr = np.array([np.array([dtype(i) for i in j[:-1].split()]) for j in x])
    return arr[:, :-1], arr[:, -1]

class KNN():
    def __init__(self, k=1):
        self.k = k
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        return self
        
    def predict_single(self, test_x):
        dis = pd.Series(euclidean_distance(self.x, test_x))
        k_nearest = dis.sort_values()[:self.k]
        k_nearest_y = self.y[k_nearest.index]
        return np.bincount(k_nearest_y).argmax()
        
    def predict(self, test_x_list):
        return np.fromiter(map(self.predict_single, test_x_list), dtype = int)
