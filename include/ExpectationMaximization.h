#pragma once

#include "../include/Eigen/Dense"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

class ExpectationMaximization
{
public:
	int number_data_dimension; // 输入数据的维度
	double epsilon_early_stop;
	bool early_stop;
	int max_iter;
	double prob_cutoff;
	
public:
	ExpectationMaximization();

	ExpectationMaximization(int n_Dims);
	
	void trainExpectationMaximization(const Eigen::MatrixXd data_block, const int& number_components, 
		Eigen::MatrixXd mu, Eigen::MatrixXd sigma, std::vector<double> component_proportion);
	
private:
	/**
	 * 计算数据流中每个数据点（具有D个维度）的隶属度（E-step: 期望步）。
	 *
	 * @param data_stream 输入数据流，表示观测数据的集合。
	 *        类型: Eigen::MatrixXd
	 *        维度: [D]，其中 D 为数据点的维度的数量。
	 *		  描述: 输入规范为行向量。
	 *
	 * @param mixing_proportion 混合比例，表示每个成分在混合分布中的权重。
	 *        类型: double
	 *        维度: [1]，标量（单个值）。
	 *
	 * @param mu 均值向量，表示高斯分布中每个成分的均值。
	 *        类型: Eigen::MatrixXd
	 *        维度: [D][K]，其中 D 为输入的某个高斯组分的维度，K 是组分的数目
	 *
	 * @param sigma 协方差矩阵，表示高斯分布中每个成分的标准差。
	 *        类型: Eigen::MatrixXd
	 *        维度: [D][D]，其中 D 为输入的某个高斯组分的维度。
	 *
	 * @return double 返回数据流中每个数据点的隶属度值。
	 *         类型: double
	 *         维度: [1]，与输入的数据流长度一致。
	 */
	Eigen::RowVectorXd estimateExpectationStep(const Eigen::MatrixXd data_block, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma);
    // 后序不能再重复的添加 引用&

	double updateComponentProportion(const Eigen::RowVectorXd& responsibility);

	Eigen::VectorXd updateMu(Eigen::MatrixXd data_block, const Eigen::RowVectorXd& responsibility);

	Eigen::MatrixXd updateSigma(const Eigen::MatrixXd data_block, const Eigen::VectorXd& mu, const Eigen::RowVectorXd& responsibility);

};

