# import time
# start = time.time()
# time.sleep(2)
# end = time.time()
# print end - start


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
X = np.array([[2, 2, 3], [2, 1, 3], [4, 1, 3]])
print euclidean_distances(X, squared=False)
# w = np.ones(X.shape[0], dtype=np.float64)
#
# print X
# print w
#
# print '''sdjlsjfl
# sfsfjls
# sfslfklfs
# '''
#
#
# print '''
#     maxw = np_points.shape[0] / (k * 100)
#     for i in xrange(length):
#         for j in xrange(k):
#             dist = np.sum((np.square(cluster_centers[j].point - np_points[i])))
#             if dist < 100:
#                 weights[i] = maxw
#     '''