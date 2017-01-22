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


def afkmc2(array, k):
    """
    afkmc2 寻找中心点seed，并聚类
    :param array: 数据集
    :param k: 中心点个数
    :return cluster_centers: 聚类后的中心点
    """
    print 'afkmc2'
    np_points = np.array(array)
    centers = kmc2.kmc2(np_points, k, afkmc2=True)
    np_centers = [list() for _ in xrange(k)]
    for index in xrange(k):
        np_centers[index] = copy(centers[index])
    cluster_centers = start_kmeans(np_points, np_centers, k)
    return cluster_centers


def kmc2_(array, k):
    """
    kmc2_ 寻找中心点seed，并聚类
    :param array: 数据集
    :param k: 中心点个数
    :return cluster_centers: 聚类后的中心点
    """
    print 'kmc2'
    np_points = np.array(array)
    centers = kmc2.kmc2(np_points, k, afkmc2=False)
    np_centers = [list() for _ in xrange(k)]
    for index in xrange(k):
        np_centers[index] = copy(centers[index])
    cluster_centers = start_kmeans(np_points, np_centers, k)
    return cluster_centers


def kpp(array, k):
    """
    kpp 寻找中心点seed，并聚类
    :param array: 数据集
    :param k: 中心点个数
    :return cluster_centers: 聚类后的中心点
    """
    print 'kpp'
    np_points = np.array(array)
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
    cluster_centers = [Point() for _ in xrange(k)]
    for index in xrange(k):
        cluster_centers[index].point = np_centers[index]
    print '开始 Kmeans'
    cluster_centers = Kmeans.kmeans_with_center(cluster_points, cluster_centers)
    for center in cluster_centers:
        print center.point
    return cluster_centers


def compare_centers(cluster_centers_afkmc2, cluster_centers_kmc2, cluster_centers_kpp, k):
    for index in xrange(k):
        if abs(cluster_centers_afkmc2[index] - cluster_centers_kmc2[index]) > 0.01 or \
                        abs(cluster_centers_afkmc2[index] - cluster_centers_kpp[index]) > 0.01:
            return False
    return True


def main(path="poker-hand-training-true.txt", split_char=','):
    k = 7  # # clusters
    array = init_array(path, split_char)
    start_afkmc2 = time.time()
    cluster_centers_afkmc2 = afkmc2(array, k)
    end_afkmc2 = time.time()
    start_kmc2 = end_afkmc2
    cluster_centers_kmc2 = kmc2_(array, k)
    end_kmc2 = time.time()
    start_kpp = end_kmc2
    cluster_centers_kpp = kpp(array, k)
    end_kpp = time.time()
    is_center_correct = compare_centers(cluster_centers_afkmc2, cluster_centers_kmc2, cluster_centers_kpp)
    during_afkmc2 = end_afkmc2 - start_afkmc2
    during_kmc2 = end_kmc2 - start_kmc2
    during_kpp = end_kpp - start_kpp
    if is_center_correct:
        print 'during_afkmc2=' + str(during_afkmc2)
        print 'during_afkmc2=' + str(during_kmc2)
        print 'during_kpp=' + str(during_kpp)
    else:
        print 'find center to kmeans fail'


if __name__ == "__main__":
    main()

