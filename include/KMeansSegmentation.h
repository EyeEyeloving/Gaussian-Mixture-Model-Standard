#pragma once

#include <iostream> 
#include <string>
#include <vector>
#include "../include/Eigen/Dense"

struct KMeansPPlus
{
	std::vector<double> component_proportion;
	Eigen::MatrixXd mu;
	Eigen::MatrixXd covariance;
};

class KMeansSegmentation
{
public:
	int number_component;
	std::string share_covariance;

private:
	std::vector<double> component_proportion;
	Eigen::MatrixXd mu;
	Eigen::MatrixXd covariance;

public:
	KMeansSegmentation();

	KMeansSegmentation(int number_component);

	void fit(Eigen::MatrixXd data_block);

	/*获取基于kmeans ++算法的初始化参数*/
	KMeansPPlus getInitialParameter() const;

private:
	Eigen::RowVectorXd distfun(const Eigen::MatrixXd& data_block,
		const Eigen::VectorXd& Centroids, const Eigen::VectorXd& var);
};

