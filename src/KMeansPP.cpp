#include "KMeansPP.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <limits>

// 构造函数，初始化k和最大迭代次数
KMeansPP::KMeansPP(int k, int max_iters) : k(k), max_iters(max_iters) {}

// 执行K-means++初始化并运行K-means聚类
void KMeansPP::fit(const Eigen::MatrixXd& data) {
    initializeCentroids(data);
    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<int> assignments = assignClusters(data);
        updateCentroids(data, assignments);
    }
}

// 获取最终质心
Eigen::MatrixXd KMeansPP::getCentroids() const {
    return centroids;
}

// 获取每个聚类的均值和协方差
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KMeansPP::getMeansAndCovariances(const Eigen::MatrixXd& data) const {
    Eigen::MatrixXd means = centroids; // 均值等于质心
    Eigen::MatrixXd covariances = Eigen::MatrixXd::Zero(data.rows(), data.rows() * k); // 协方差矩阵

    for (int i = 0; i < k; ++i) {
        Eigen::MatrixXd cluster_data(data.rows(), data.cols());
        int cluster_size = 0;

        for (int j = 0; j < data.cols(); ++j) {
            if (assignClusters(data)[j] == i) {
                cluster_data.col(cluster_size) = data.col(j);
                cluster_size++;
            }
        }

        if (cluster_size > 0) {
            cluster_data = cluster_data.leftCols(cluster_size);  // 截取有效的列
            Eigen::VectorXd mean = cluster_data.rowwise().mean();
            covariances.block(0, i * data.rows(), data.rows(), data.rows()) =
                (cluster_data.colwise() - mean) * (cluster_data.colwise() - mean).transpose() / cluster_size;
        }
    }

    return { means, covariances };
}

// 初始化质心 (K-means++)
void KMeansPP::initializeCentroids(const Eigen::MatrixXd& data) {
    int n_samples = data.cols();  // 数据点的数量
    int n_features = data.rows();  // 特征维度
    centroids = Eigen::MatrixXd(n_features, k);  // 质心矩阵 D x K

    // 随机选择第一个质心
    srand(static_cast<unsigned int>(time(0)));  // 设置随机种子
    int first_centroid_index = rand() % n_samples;
    centroids.col(0) = data.col(first_centroid_index);

    // 依次选择其余的质心
    for (int c = 1; c < k; ++c) {
        Eigen::VectorXd distances = calculateMinDistances(data);
        double total_distance = distances.sum();
        double r = ((double)rand() / RAND_MAX) * total_distance;

        double cumulative_distance = 0;
        int chosen_index = 0;
        for (int i = 0; i < n_samples; ++i) {
            cumulative_distance += distances(i);
            if (cumulative_distance >= r) {
                chosen_index = i;
                break;
            }
        }
        centroids.col(c) = data.col(chosen_index);
    }
}

// 计算每个点到最近质心的最小距离的平方
Eigen::VectorXd KMeansPP::calculateMinDistances(const Eigen::MatrixXd& data) const {
    int n_samples = data.cols();  // 数据点的数量
    Eigen::VectorXd min_distances(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        double min_distance = std::numeric_limits<double>::max();
        for (int j = 0; j < centroids.cols(); ++j) {
            double distance = (data.col(i) - centroids.col(j)).squaredNorm();
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
        min_distances(i) = min_distance;
    }
    return min_distances;
}

// 将每个数据点分配给最近的质心
std::vector<int> KMeansPP::assignClusters(const Eigen::MatrixXd& data) const {
    int n_samples = data.cols();  // 数据点的数量
    std::vector<int> assignments(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int closest_centroid = 0;
        for (int j = 0; j < centroids.cols(); ++j) {
            double distance = (data.col(i) - centroids.col(j)).squaredNorm();
            if (distance < min_distance) {
                min_distance = distance;
                closest_centroid = j;
            }
        }
        assignments[i] = closest_centroid;
    }
    return assignments;
}

// 更新质心
void KMeansPP::updateCentroids(const Eigen::MatrixXd& data, const std::vector<int>& assignments) {
    int n_features = data.rows();  // 特征维度
    Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(n_features, k);
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(k);

    for (int i = 0; i < data.cols(); ++i) {
        int cluster_id = assignments[i];
        new_centroids.col(cluster_id) += data.col(i);
        counts(cluster_id)++;
    }

    for (int j = 0; j < k; ++j) {
        if (counts(j) > 0) {
            new_centroids.col(j) /= counts(j);
        }
    }
    centroids = new_centroids;
}
