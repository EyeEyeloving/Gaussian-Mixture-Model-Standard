#include "GaussianMixture.h"
#include "KMeansPP.h"

GaussianMixture::GaussianMixture()
	: number_data_dimension(1), number_components(5), early_stop(false), max_iter(100) {
	component_proportion.resize(1, 5.0);
	mu.resize(1, 5);
	sigma.resize(1, 1);
}

GaussianMixture::GaussianMixture(int& n_Comps)
	: number_data_dimension(1), number_components(n_Comps), early_stop(false), max_iter(100) {
	
}

void GaussianMixture::validateDataInput(Eigen::MatrixXd& data_block) {
	/*输入是引用/指针就可以是void输出*/
	if (data_block.rows() > data_block.cols()) data_block = data_block.transpose(); // 把输入数据转化为[D][N]
	if (data_block.rows() == 1) data_block = data_block.transpose(); // 如果输入是一个样本（默认样本维度不为1）
	std::cout << "Validate the Input Data >>" << std::endl;
	std::cout << "The Input Size is: " << data_block.cols() << std::endl;
	std::cout << "The Input Dimension is: " << data_block.rows() << std::endl;
}

gmModel GaussianMixture::trainGaussianMixture(Eigen::MatrixXd& data_block) {

	/*处理输入数据*/
	validateDataInput(data_block);
	number_data_dimension = data_block.cols(); // 数据的维度
	number_data = data_block.rows(); // 数据量

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
		sigma += covariances.block(0, j * data_block.rows(), data_block.rows(), data_block.rows());
	}

	/*训练*/
	trainExpectationMaximization(data_block, number_components, mu, sigma, component_proportion);

	return { component_proportion, mu, sigma };
}