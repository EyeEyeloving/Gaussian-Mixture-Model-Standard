#include "GaussianMixture.h"
#include "KMeansPP.h"

GaussianMixture::GaussianMixture()
	: number_data_dimension(1), number_data(1), number_components(5), early_stop(false), max_iter(100) {
	component_proportion.resize(1, 5.0);
	mu.resize(1, 5);
	sigma.resize(1, 1);
}

GaussianMixture::GaussianMixture(int& n_Comps)
	: number_data_dimension(1), number_data(1), number_components(n_Comps), early_stop(false), max_iter(100) {
	
}

Eigen::MatrixXd GaussianMixture::validateDataInput(Eigen::MatrixXd& data_block) {
	if (data_block.rows() > data_block.cols()) return data_block.transpose(); // 把输入数据转化为[D][N]
	// 原代码data_block = data_block.transpose()存在data_block转置后维度与自身不相符的问题

	/*输入是引用/指针就可以是void输出*/
}

gmModel GaussianMixture::trainGaussianMixture(Eigen::MatrixXd& data_block_raw) {

	/*处理输入数据*/
	Eigen::MatrixXd data_block = validateDataInput(data_block_raw);
	number_data_dimension = data_block.rows(); // 数据的维度
	number_data = data_block.cols(); // 数据量

	/*定义GMM参数*/
	component_proportion.resize(number_components, 1/number_components);
	mu.resize(number_data_dimension, number_components);
	sigma.resize(number_data_dimension, number_data_dimension);
	
	/*基于KMeans++方法的初始化*/
	KMeansPP kmeanspp(number_components, 100);
	kmeanspp.fit(data_block);
	auto output = kmeanspp.getMeansAndCovariances(data_block);
	mu = output.first;
	auto covariances = output.second;
	for (int j = 0; j < number_components; j++) {
		sigma += (1 / number_components * covariances.block(0, j * data_block.rows(), data_block.rows(), data_block.rows()));
	}

	/* 打印 mu 和 sigma 的值 */
	std::cout << "Mu (Means):\n" << mu << std::endl;
	std::cout << "Sigma (Covariances):\n" << sigma << std::endl;

	/*训练*/
	trainExpectationMaximization(data_block, number_components, mu, sigma, component_proportion);

	return { component_proportion, mu, sigma };
}