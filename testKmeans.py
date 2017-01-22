# -*- coding: utf-8 -*-
import numpy as np
import kmc2
from Kmeans import Point
import Kmeans
from copy import copy


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
    :return: 无返回值
    """
    np_points = np.array(array)
    centers = kmc2.kmc2(np_points, k, afkmc2=True)
    length = len(centers)
    np_centers = [list() for _ in xrange(length)]
    for index in xrange(length):
        np_centers[index] = copy(centers[index])
    start_kmeans(np_points, np_centers)


def kmc2_(array, k):
    """
    kmc2_ 寻找中心点seed，并聚类
    :param array: 数据集
    :param k: 中心点个数
    :return: 无返回值
    """
    np_points = np.array(array)
    centers = kmc2.kmc2(np_points, k, afkmc2=False)
    length = len(centers)
    np_centers = [list() for _ in xrange(length)]
    for index in xrange(length):
        np_centers[index] = copy(centers[index])
    start_kmeans(np_points, np_centers)


def kpp(array, k):
    """
    kpp 寻找中心点seed，并聚类
    :param array: 数据集
    :param k: 中心点个数
    :return: 无返回值
    """
    np_points = np.array(array)
    np_centers = Kmeans.kpp(np_points, k)
    start_kmeans(np_points, np_centers)


def start_kmeans(np_points, np_centers):
    print '中心点 seeding'
    print np_centers
    cluster_points = [Point() for _ in xrange(len(np_points))]
    print '将numpy 数组 转化为 多维空间 point'
    for index in xrange(len(np_points)):
        cluster_points[index].point = np_points[index]
    cluster_centers = [Point() for _ in xrange(len(np_centers))]
    for index in xrange(len(np_centers)):
        cluster_centers[index].point = np_centers[index]
    Kmeans.point2center(cluster_points, cluster_centers)
    print '开始 Kmeans'
    cluster_centers = Kmeans.kmeans_with_center(cluster_points, cluster_centers)
    for center in cluster_centers:
        print center.point


def main(path="poker-hand-training-true.txt", split_char=','):
    k = 7  # # clusters
    array = init_array(path, split_char)
    afkmc2(array, k)
    kmc2_(array, k)
    kpp(array, k)


if __name__ == "__main__":
    main()

