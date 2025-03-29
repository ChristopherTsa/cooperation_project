import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

with open('C:/Users/Thibaud/3A/cooperation_project/data/first_database.pkl', 'rb') as f:
    x,y = pickle.load(f)

#visualisation sur un graphique

#print(type(x), x.shape)
#print(type(y), y.shape)

#plt.scatter(x, y, c=y)
#plt.show()

sel = [i for i in range(n)]
ind = np.random.choice(sel, m, replace=False)
x_selected = [x[i] for i in ind]
