# -*- coding: utf-8 -*-
from copy import copy
from random import random, choice
try:
    import psyco

    psyco.full()
except ImportError:
    pass

FLOAT_MAX = 1e100


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


def nearest_cluster_center(point, cluster_centers):
    """Distance and index of the closest cluster center"""
    def sqr_distance(a, b):
        dist = 0.0
        length = len(a.point)
        for index in xrange(length):
            dist += (a.point[index] - b.point[index]) ** 2
        return dist
    min_index = point.group
    min_dist = FLOAT_MAX
    for i, cc in enumerate(cluster_centers):
        d = sqr_distance(cc, point)
        if min_dist > d:
            min_dist = d
            min_index = i

    return min_index, min_dist


def nearest_np_center(np_point, np_centers):
    """Distance and index of the closest cluster center"""
    def sqr_distance(a, b):
        dist = 0.0
        length = len(a)
        for index in xrange(length):
            dist += (a[index] - b[index]) ** 2
        return dist

    min_dist = FLOAT_MAX
    for i, cc in enumerate(np_centers):
        d = sqr_distance(cc, np_point)
        if min_dist > d:
            min_dist = d

    return min_dist


def kmeans_with_center(cluster_points, cluster_centers):
    """
    根据参数的 点集 和 中心点 进行Kmeans聚类
    :param cluster_points:  点集
    :param cluster_centers:  初始中心点
    :return: cluster_centers 中心点
    """
    print '划定聚类中心'
    for np_point in cluster_points:
        np_point.group = nearest_cluster_center(np_point, cluster_centers)[0]
    # 迭代阈值
    lenpts10 = len(cluster_points) >> 10
    print '开始迭代'
    while True:
        # group element for centroids are used as counters
        # 初始化中心点
        length = len(cluster_centers[0].point)
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
        if changed <= lenpts10:
            break
    # TODO 重新划分一下组号，有必要么？是不是要更新每个点的组号呢？
    for i, cc in enumerate(cluster_centers):
        cc.group = i

    return cluster_centers


def kpp(np_points, k):
    """
    kpp 寻找中心点seed
    :param np_points:
    :param k:
    :return:
    """
    np_centers = [list() for _ in xrange(k)]
    np_centers[0] = copy(choice(np_points))
    d = [0.0 for _ in xrange(len(np_points))]
    for i in xrange(1, len(np_centers)):
        result = 0
        for j, cluster_point in enumerate(np_points):
            d[j] = nearest_np_center(cluster_point, np_centers[:i])
            result += d[j]
        result *= random()
        for j, di in enumerate(d):
            result -= di
            if result > 0:
                continue
            np_centers[i] = copy(np_points[j])
            break
    return np_centers


def point2center(cluster_points, cluster_centers):
    """
    将点集指向最近的中心点
    :param cluster_points:  点集
    :param cluster_centers: 中心点
    :return: 不需要返回值
    """
    for cluster_point in cluster_points:
        cluster_point.group = nearest_cluster_center(cluster_point, cluster_centers)[0]
