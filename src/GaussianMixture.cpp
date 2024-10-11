#include "GaussianMixture.h"
// #include "KMeansPP.h"
#include "KMeansSegmentation.h"

GaussianMixture::GaussianMixture()
	: number_data_dimension(1), number_data(1), number_components(5), early_stop(false), max_iter(100) {
	component_proportion.resize(5, 1.0 / 5);
	mu.resize(1, 5);
	sigma.resize(1, 1);
}

GaussianMixture::GaussianMixture(int& n_Comps)
	: number_data_dimension(1), number_data(1), number_components(n_Comps), early_stop(false), max_iter(100) {
	
}

Eigen::MatrixXd GaussianMixture::validateDataInput(Eigen::MatrixXd& data_block) {
	if (data_block.rows() > data_block.cols()) return data_block.transpose(); // 把输入数据转化为[D][N]
	// 原代码data_block = data_block.transpose()存在data_block转置后维度与自身不相符的问题
}

void GaussianMixture::normalizeDataInput(Eigen::MatrixXd& data_block) {
	/*输入是引用/指针就可以是void输出*/
	Eigen::RowVectorXd min_col_value = data_block.colwise().minCoeff(); // 每一列的最小值
	double min_value = min_col_value.minCoeff();
	Eigen::RowVectorXd max_col_value = data_block.colwise().maxCoeff(); // 每一列的最大值
	double max_value = max_col_value.maxCoeff();

	data_block = (data_block.array() - min_value) / (max_value - min_value);
	//data_block = (data_block.colwise() - min_col_value) / (max_col_value - min_col_value);
	// 元素操作的广播问题：Eigen::MatrixXd 和 Eigen::VectorXd 之间不支持直接的减法或除法
	// 矩阵和向量之间的广播机制不是默认支持的
	// 在使用 .array() 后，后续的运算将基于数组而不是矩阵运算，这可能会导致维度不匹配的问题。
}

gmModel GaussianMixture::trainGaussianMixture(Eigen::MatrixXd& data_block_raw) {

	/*处理输入数据*/
	Eigen::MatrixXd data_block = validateDataInput(data_block_raw);
	normalizeDataInput(data_block);
	number_data_dimension = data_block.rows(); // 数据的维度
	number_data = data_block.cols(); // 数据量

	/*定义GMM参数*/
	component_proportion.resize(number_components, 1.0 / number_components);
	mu.resize(number_data_dimension, number_components);
	sigma.resize(number_data_dimension, number_data_dimension);
	sigma.setZero(); // 否则，sigma会自动赋值一个很大的随机值
	
	//*基于KMeans++方法的初始化*/

	KMeansSegmentation kmeanspp(number_components);
	kmeanspp.fit(data_block);
	auto init_paras = kmeanspp.getInitialParameter();
	mu = init_paras.mu;
	sigma = init_paras.covariance;

	//KMeansPP kmeanspp(number_components, 100, 1e-3);
	//kmeanspp.fit(data_block);
	//auto output = kmeanspp.getMeansAndCovariances(data_block);
	//mu = output.first;
	//auto& covariances = output.second;
	//for (int j = 0; j < number_components; j++) {
	//	std::cout << covariances.block(0, j * data_block.rows(), data_block.rows(), data_block.rows()) << std::endl;
	//	sigma += (1.0 / number_components * covariances.block(0, j * data_block.rows(), data_block.rows(), data_block.rows())); 
	//	// 1/number是int类型的除法
	//	// sigma += (1 / number_components * covariances.block(0, j * data_block.rows(), data_block.rows(), data_block.rows()));
	//}

	/* 打印 mu 和 sigma 的值 */
	std::cout << "Mu (Means):\n" << mu << std::endl;
	std::cout << "Sigma (Covariances):\n" << sigma << std::endl;

	/*训练*/
	trainExpectationMaximization(data_block, number_components, mu, sigma, component_proportion);

	return { component_proportion, mu, sigma };
}