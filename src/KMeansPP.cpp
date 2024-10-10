/*Generated and Debuged by GPT*/

#include "KMeansPP.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <limits>

//// 构造函数，初始化k和最大迭代次数
//KMeansPP::KMeansPP(int k, int max_iters) : k(k), max_iters(max_iters) {}
//
//// 执行K-means++初始化并运行K-means聚类
//void KMeansPP::fit(const Eigen::MatrixXd& data) {
//    initializeCentroids(data);
//    for (int iter = 0; iter < max_iters; ++iter) {
//        std::vector<int> assignments = assignClusters(data);
//        updateCentroids(data, assignments);
//        std::cout << "Iteration Step in Initialization kmeans++ Method: " << iter + 1 << std::endl;
//    }
//    std::cout << "KMeansPP Method.fit() completed" << std::endl;
//}

// 构造函数，初始化k，最大迭代次数，以及容限
KMeansPP::KMeansPP(int k, int max_iters, double tolerance) : k(k), max_iters(max_iters), tolerance(tolerance) {}

// 执行K-means++初始化并运行K-means聚类
void KMeansPP::fit(const Eigen::MatrixXd& data) {
    initializeCentroids(data);
    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<int> assignments = assignClusters(data);
        Eigen::MatrixXd old_centroids = centroids;  // 保存旧的质心
        updateCentroids(data, assignments);

        // 计算质心的变化
        double centroid_shift = (centroids - old_centroids).norm();
        std::cout << "Iteration Step: " << iter + 1 << ", Centroid shift: " << centroid_shift << std::endl;

        // 检查是否提前停止
        if (centroid_shift < tolerance) {
            std::cout << "Early stopping at iteration " << iter + 1 << " due to centroid shift below tolerance." << std::endl;
            break;
        }
    }
    std::cout << "KMeansPP Method.fit() completed" << std::endl;
}

// 获取最终质心
Eigen::MatrixXd KMeansPP::getCentroids() const {
    return centroids;
}

// 获取每个聚类的均值和协方差
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KMeansPP::getMeansAndCovariances(const Eigen::MatrixXd& data) const {
    Eigen::MatrixXd means = centroids; // 簇均值等于质心
    Eigen::MatrixXd covariances = Eigen::MatrixXd::Zero(data.rows(), data.rows() * k); // 初始化协方差矩阵

    // 获取所有点的簇分配结果
    std::vector<int> cluster_assignments = assignClusters(data);

    // 遍历每个簇
    for (int j = 0; j < k; ++j) {
        std::cout << "Processing cluster " << j + 1 << " of " << k << "..." << std::endl;
        // 找到属于簇j的所有点的索引
        std::vector<int> indices;
        for (int i = 0; i < cluster_assignments.size(); ++i) {
            if (cluster_assignments[i] == j) {
                indices.push_back(i);
            }
        }

        int cluster_size = indices.size();  // 簇的大小
        std::cout << "Cluster size: " << cluster_size << " points." << std::endl;

        if (cluster_size > 0) {
            // 提取属于该簇的所有数据列
            Eigen::MatrixXd cluster_data(data.rows(), cluster_size);
            for (int i = 0; i < cluster_size; ++i) {
                cluster_data.col(i) = data.col(indices[i]);
            }

            // 计算该簇的均值：“簇的大小”表示被分配到某个特定簇的样本点的数量。
            Eigen::VectorXd mean = cluster_data.rowwise().mean();
            std::cout << "Cluster " << j + 1 << " mean calculated." << std::endl;

            // 计算协方差矩阵
            Eigen::MatrixXd centered = cluster_data.colwise() - mean;
            covariances.block(0, j * data.rows(), data.rows(), data.rows()) =
                (centered * centered.transpose()) / cluster_size;
            std::cout << "Cluster " << j + 1 << " covariance matrix calculated." << std::endl;
        }
    }

    std::cout << "Means and covariances calculation completed." << std::endl;
    return { means, covariances };
}


// 初始化质心 (K-means++)
void KMeansPP::initializeCentroids(const Eigen::MatrixXd& data) {
    int n_samples = static_cast<int>(data.cols());  // 数据点的数量
    int n_features = static_cast<int>(data.rows());  // 特征维度
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
    int n_samples = static_cast<int>(data.cols());  // 数据点的数量
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
    int n_samples = static_cast<int>(data.cols());  // 数据点的数量
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
    int n_features = static_cast<int>(data.rows());  // 特征维度
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
