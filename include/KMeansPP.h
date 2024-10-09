#pragma once

#include <Eigen/Dense>
#include <vector>

class KMeansPP {
public:
    // 构造函数，接受聚类数量k和最大迭代次数max_iters
    KMeansPP(int k, int max_iters);

    // 运行K-means++初始化和K-means聚类
    void fit(const Eigen::MatrixXd& data);

    // 获取最终质心
    Eigen::MatrixXd getCentroids() const;

    // 获取每个聚类的均值和协方差
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> getMeansAndCovariances(const Eigen::MatrixXd& data) const;

private:
    int k;  // 聚类数
    int max_iters;  // 最大迭代次数
    Eigen::MatrixXd centroids;  // 质心

    // 初始化质心 (K-means++)
    void initializeCentroids(const Eigen::MatrixXd& data);

    // 计算每个点到最近质心的最小距离的平方
    Eigen::VectorXd calculateMinDistances(const Eigen::MatrixXd& data) const;

    // 将每个数据点分配给最近的质心
    std::vector<int> assignClusters(const Eigen::MatrixXd& data) const;

    // 更新质心
    void updateCentroids(const Eigen::MatrixXd& data, const std::vector<int>& assignments);
};