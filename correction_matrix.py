import json
import numpy as np
import statsmodels.api as sm
from pandas.stats.api import ols
import pandas as pd
import statsmodels.formula.api as smf
from scipy.linalg import toeplitz
import scipy.spatial.distance as distance
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.optimize import curve_fit
x = np.random.rand(100,20)
y =  np.random.rand(100,20)

p = Ridge()
p.fit(x,y)

# A, resids, rank, st = np.linalg.lstsq(x, y,-10)
# print A.shape
# # print resids
# # print rank
# # print st
# avg = 0
# avg2 =0
# l = len(x) - 1
# for i in xrange(l):
#     xd1 = 1 - distance.cosine(x[i],x[i+1])
#     y_p = np.dot(x[i], A)
#     y_p2 = np.dot(x[i+1], A)
#     xd2 = 1 - distance.cosine(y_p,y_p2)
#     avg2 += xd1 - xd2
#     dist = 1 - distance.cosine(y[i],y_p)
#     avg += dist
# print(avg/l)
# print(avg2/l)
x_new = p.predict(x)
avg = 0
for i in xrange(len(x_new)):
    dist = 1 - distance.cosine(y[i],x_new[i])
    avg += dist
print(avg/len(x_new))
