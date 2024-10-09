#pragma once

#include <Eigen/Dense>
#include <vector>

class KMeansPP {
public:
    // ���캯�������ܾ�������k������������max_iters
    KMeansPP(int k, int max_iters);

    // ����K-means++��ʼ����K-means����
    void fit(const Eigen::MatrixXd& data);

    // ��ȡ��������
    Eigen::MatrixXd getCentroids() const;

    // ��ȡÿ������ľ�ֵ��Э����
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> getMeansAndCovariances(const Eigen::MatrixXd& data) const;

private:
    int k;  // ������
    int max_iters;  // ����������
    Eigen::MatrixXd centroids;  // ����

    // ��ʼ������ (K-means++)
    void initializeCentroids(const Eigen::MatrixXd& data);

    // ����ÿ���㵽������ĵ���С�����ƽ��
    Eigen::VectorXd calculateMinDistances(const Eigen::MatrixXd& data) const;

    // ��ÿ�����ݵ��������������
    std::vector<int> assignClusters(const Eigen::MatrixXd& data) const;

    // ��������
    void updateCentroids(const Eigen::MatrixXd& data, const std::vector<int>& assignments);
};