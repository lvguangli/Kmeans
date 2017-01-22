# -*- coding: utf-8 -*-
import numpy as np
import kmc2
from Kmeans import Point
import Kmeans
from copy import copy
import time


def init_array(path, split_char):
    """
    初始化数据集
    :param path:  数据集文件路径
    :param split_char: 分隔符
    :return:
    """
    print '初始化数组'
    array = list()
    with open(path, 'rw') as input_file:
        for line in input_file:
            words = line.split(split_char)
            point = list()
            for word in words:
                point.append(float(word))
            array.append(point)
    return array


def afkmc2(np_points, k):
    """
    afkmc2 寻找中心点seed，并聚类
    :param np_points: 数据集
    :param k: 中心点个数
    :return cluster_centers: 聚类后的中心点
    """
    print 'afkmc2'
    centers = kmc2.kmc2(np_points, k, afkmc2=True)
    np_centers = [list() for _ in xrange(k)]
    for index in xrange(k):
        np_centers[index] = copy(centers[index])
    cluster_centers = start_kmeans(np_points, np_centers, k)
    return cluster_centers


def kmc2_(np_points, k):
    """
    kmc2_ 寻找中心点seed，并聚类
    :param np_points: 数据集
    :param k: 中心点个数
    :return cluster_centers: 聚类后的中心点
    """
    print 'kmc2'

    centers = kmc2.kmc2(np_points, k, afkmc2=False)
    np_centers = [list() for _ in xrange(k)]
    for index in xrange(k):
        np_centers[index] = copy(centers[index])
    cluster_centers = start_kmeans(np_points, np_centers, k)
    return cluster_centers


def kpp(np_points, k):
    """
    kpp 寻找中心点seed，并聚类
    :param np_points: 数据集
    :param k: 中心点个数
    :return cluster_centers: 聚类后的中心点
    """
    print 'kpp'
    np_centers = Kmeans.kpp(np_points, k)
    cluster_centers = start_kmeans(np_points, np_centers, k)
    return cluster_centers


def start_kmeans(np_points, np_centers, k):
    print '中心点 seeding'
    print np_centers
    cluster_points = [Point() for _ in xrange(len(np_points))]
    print '将numpy 数组 转化为 多维空间 point'
    for index in xrange(len(np_points)):
        cluster_points[index].point = np_points[index]
        # cluster_points[index].group = 0 # 不需要是因为 kmeans时会预处理
    cluster_centers = [Point() for _ in xrange(k)]
    for index in xrange(k):
        cluster_centers[index].point = np_centers[index]
        cluster_centers[index].group = 0
    print '开始 Kmeans'
    cluster_centers = Kmeans.kmeans_with_center(cluster_points, cluster_centers)
    for center in cluster_centers:
        print center.point
    return cluster_centers


def qe(np_points, np_centers):
    """Compute the quantization error"""
    a1 = np.sum(np.power(np_points, 2), axis=1)
    a2 = np.dot(np_points, np_centers.T)
    a3 = np.sum(np.power(np_centers, 2), axis=1)
    dist = - 2*a2 + a3[np.newaxis, :]
    mindist = np.min(dist, axis=1) + a1
    error = np.sum(mindist)
    return error


def point2numpy(cluster_centers):
    result = list()
    for cluster_center in cluster_centers:
        result.append(cluster_center.point)
    return np.array(result)


def main(path="poker-hand-training-true.txt", split_char=','):
    k = 7  # # clusters
    array = init_array(path, split_char)
    np_points = np.array(array)
    # afkmc2
    start_afkmc2 = time.time()
    cluster_centers_afkmc2 = afkmc2(np_points, k)
    end_afkmc2 = time.time()
    #  kmc2
    start_kmc2 = end_afkmc2
    cluster_centers_kmc2 = kmc2_(np_points, k)
    end_kmc2 = time.time()
    #  kpp
    start_kpp = end_kmc2
    cluster_centers_kpp = kpp(np_points, k)
    end_kpp = time.time()
    # qe
    np_centers_afkmc2 = point2numpy(cluster_centers_afkmc2)
    np_centers_kmc2 = point2numpy(cluster_centers_kmc2)
    np_centers_kpp = point2numpy(cluster_centers_kpp)
    error_afkmc2 = qe(np_points=np_points, np_centers=np_centers_afkmc2)
    error_kmc2 = qe(np_points=np_points, np_centers=np_centers_kmc2)
    error_kpp = qe(np_points=np_points, np_centers=np_centers_kpp)
    #  during
    during_afkmc2 = end_afkmc2 - start_afkmc2
    during_kmc2 = end_kmc2 - start_kmc2
    during_kpp = end_kpp - start_kpp

    print 'error_afkmc2 = ' + str(error_afkmc2) + '    during_afkmc2 = ' + str(during_afkmc2)
    print 'error_kmc2   = ' + str(error_kmc2) + '    during_afkmc2 = ' + str(during_kmc2)
    print 'error_kpp    = ' + str(error_kpp) + '    during_kpp = ' + str(during_kpp)


if __name__ == "__main__":
    main()

