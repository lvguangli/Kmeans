# -*- coding: utf-8 -*-
from copy import copy
from random import random, choice
import numpy as np
try:
    import psyco

    psyco.full()
except ImportError:
    pass

FLOAT_MAX = 1e100
ITERATION_LOG = True


class Point:
    """
    类Point主要用于Kmeans时，其中的元素
    point是一个list，用于寻找中心点
    group是一个整数，表示该点缩归属的中心下标
    """
    __slots__ = ["point", "group"]

    def __init__(self):
        self.group = 0
        self.point = list()


def nearest_cluster_center(cluster_point, cluster_centers):
    """Distance and index of the closest cluster center"""
    min_index = cluster_point.group
    min_dist = FLOAT_MAX
    for i, cluster_center in enumerate(cluster_centers):
        d = np.sum((np.square(cluster_center.point - cluster_point.point)))
        if min_dist > d:
            min_dist = d
            min_index = i
    return min_index, min_dist


def nearest_np_center_dist(np_point, np_centers):
    """Distance and index of the closest cluster center"""
    min_dist = FLOAT_MAX
    for i, np_center in enumerate(np_centers):
        d = np.sum((np.square(np_center - np_point)))
        if min_dist > d:
            min_dist = d
    return min_dist


def kmeans_with_center(cluster_points, cluster_centers, times):
    """
    根据参数的 点集 和 中心点 进行Kmeans聚类
    :param cluster_points:  点集
    :param cluster_centers:  初始中心点
    :param times:  迭代的最大次数  0 没有限制
    :return: cluster_centers 中心点
    """
    print '划定聚类中心'
    for cluster_point in cluster_points:
        cluster_point.group = nearest_cluster_center(cluster_point, cluster_centers)[0]
    # 迭代阈值
    lenpts10 = len(cluster_points) >> 10
    length = len(cluster_centers[0].point)
    print '开始迭代'
    count = 0
    while True:
        if times != 0 and count >= times:
            break
        count += 1
        if ITERATION_LOG:
            print '一次迭代开始' + ' count=' + str(count)
        # group element for centroids are used as counters
        # 初始化中心点
        for cc in cluster_centers:
            cc.point = [0.0 for _ in xrange(length)]
            cc.group = 0
        # 对每个点划分中心
        for np_point in cluster_points:
            cluster_centers[np_point.group].group += 1
            for index in xrange(length):
                cluster_centers[np_point.group].point[index] += np_point.point[index]
        for cc in cluster_centers:
            for index in xrange(length):
                cc.point[index] /= cc.group
        # find closest centroid of each PointPtr
        # 检查所有点是否仍属于原来的中心，并计数
        changed = 0
        for np_point in cluster_points:
            min_i = nearest_cluster_center(np_point, cluster_centers)[0]
            if min_i != np_point.group:
                changed += 1
                np_point.group = min_i
        # stop when 99.9% of points are good
        if ITERATION_LOG:
            print 'changed=' + str(changed)
            print 'lenpts10=' + str(lenpts10)
            print '一次迭代结束'
        if changed <= lenpts10:
            break
    # 重新划分一下组号
    for i, cc in enumerate(cluster_centers):
        cc.group = i
    return cluster_centers


def kpp(np_points, k):
    """
    kpp 寻找中心点seed
    :param np_points: 点集
    :param k: 中心数
    :return: 聚类中心 seed
    """
    np_centers = [list() for _ in xrange(k)]
    np_centers[0] = copy(choice(np_points))
    d = [0.0 for _ in xrange(len(np_points))]
    for i in xrange(1, k):
        result = 0
        for j, cluster_point in enumerate(np_points):
            d[j] = nearest_np_center_dist(cluster_point, np_centers[:i])
            result += d[j]
        result *= random()
        for j, di in enumerate(d):
            result -= di
            if result > 0:
                continue
            np_centers[i] = copy(np_points[j])
            break
    return np_centers

