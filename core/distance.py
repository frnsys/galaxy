import numpy as np
from scipy.spatial.distance import cosine
def custom_dist(X, Y):
    X_pub = X[1]
    X_bow = X[:101]
    X_con = X[101:]

    Y_pub = Y[1]
    Y_bow = Y[:101]
    Y_con = Y[101:]


    weights = [1, 100, 1]
    dist_bow = cosine(X_bow, Y_bow)
    dist_con = cosine(X_con, Y_con)
    dist_pub = abs(X_pub - Y_pub)

    total = sum([weights[i]*d for i, d in enumerate([dist_bow, dist_con, dist_pub])])
    #print('dist_bow: {0}, dist_con: {1}, dist_pub: {2}'.format(dist_bow, dist_con, dist_pub))

    val = total/sum(weights)
    return val

