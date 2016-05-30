import numpy as np
import statsmodels.api as sm
from scipy.linalg import toeplitz
import scipy.spatial.distance as distance

def test(matrix,x,y):
    distances = np.zeros(len(x))
    trans = np.dot(x,matrix)
    losses = np.zeros(len(x)-1)
    l = len(x)
    for i in xrange(l):
        dist = 1 - distance.cosine(y[i],trans[i])
        distances[i] = dist
    for i in xrange(l -1):
        dist1 = distance.cosine(x[i],x[i+1])
        dist2 = distance.cosine(trans[i],trans[i+1])
        losses[i] = dist1 - dist2

    print('avg', distances.mean())
    print('std', distances.std())
    print('best', distances.max())
    print('worst', distances.min())
    print('loss', losses.mean())
    return distances,losses

# x = np.random.rand(100000,20)
# y =  np.random.rand(100000,20)
# A = np.empty((0,y.shape[1]))
# print(A.shape
# for i in xrange(y.shape[1]):
#     ols_resid = sm.OLS(y[:,i], x).fit().resid
#     resid_fit = sm.OLS(ols_resid[1:], sm.add_constant(ols_resid[:-1])).fit()
#     rho = resid_fit.params[1]
#     order = toeplitz(range(len(ols_resid)))
#
#     sigma = rho**order
#     gls_model = sm.GLS(y[:,i], x, sigma=sigma)
#     gls_results = gls_model.fit()
#     A = np.append(A,np.array([gls_results.params]), axis=0)
#     print(i
# print(A.shape
# print('GLS true'
# test(A,x,y)

# print('OLS'
# A = sm.OLS(y, x).fit().params
# test(A,x,y)
#
# print('numpy OLS'
# A, resids, rank, st = np.linalg.lstsq(x, y)
# test(A,x,y)
#
# print('GLSAR'
# A = np.empty((0,y.shape[1]))
def GLSAR(x,y):
    A = np.zeros((y.shape[1],y.shape[1]))
    for i in xrange(y.shape[1]):
        glsar_model = sm.GLSAR(y[:,i], x, 0.5)
        glsar_results = glsar_model.iterative_fit()
        A[i] = glsar_results.params
        print(i)
    # print(A.shape
    test(A, x, y)
    return A

