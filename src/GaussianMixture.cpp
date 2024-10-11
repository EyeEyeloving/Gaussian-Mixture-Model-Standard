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
	if (data_block.rows() > data_block.cols()) return data_block.transpose(); // ����������ת��Ϊ[D][N]
	// ԭ����data_block = data_block.transpose()����data_blockת�ú�ά�����������������
}

void GaussianMixture::normalizeDataInput(Eigen::MatrixXd& data_block) {
	/*����������/ָ��Ϳ�����void���*/
	Eigen::RowVectorXd min_col_value = data_block.colwise().minCoeff(); // ÿһ�е���Сֵ
	double min_value = min_col_value.minCoeff();
	Eigen::RowVectorXd max_col_value = data_block.colwise().maxCoeff(); // ÿһ�е����ֵ
	double max_value = max_col_value.maxCoeff();

	data_block = (data_block.array() - min_value) / (max_value - min_value);
	//data_block = (data_block.colwise() - min_col_value) / (max_col_value - min_col_value);
	// Ԫ�ز����Ĺ㲥���⣺Eigen::MatrixXd �� Eigen::VectorXd ֮�䲻֧��ֱ�ӵļ��������
	// ���������֮��Ĺ㲥���Ʋ���Ĭ��֧�ֵ�
	// ��ʹ�� .array() �󣬺��������㽫������������Ǿ������㣬����ܻᵼ��ά�Ȳ�ƥ������⡣
}

gmModel GaussianMixture::trainGaussianMixture(Eigen::MatrixXd& data_block_raw) {

	/*������������*/
	Eigen::MatrixXd data_block = validateDataInput(data_block_raw);
	normalizeDataInput(data_block);
	number_data_dimension = data_block.rows(); // ���ݵ�ά��
	number_data = data_block.cols(); // ������

	/*����GMM����*/
	component_proportion.resize(number_components, 1.0 / number_components);
	mu.resize(number_data_dimension, number_components);
	sigma.resize(number_data_dimension, number_data_dimension);
	sigma.setZero(); // ����sigma���Զ���ֵһ���ܴ�����ֵ
	
	//*����KMeans++�����ĳ�ʼ��*/

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
	//	// 1/number��int���͵ĳ���
	//	// sigma += (1 / number_components * covariances.block(0, j * data_block.rows(), data_block.rows(), data_block.rows()));
	//}

	/* ��ӡ mu �� sigma ��ֵ */
	std::cout << "Mu (Means):\n" << mu << std::endl;
	std::cout << "Sigma (Covariances):\n" << sigma << std::endl;

	/*ѵ��*/
	trainExpectationMaximization(data_block, number_components, mu, sigma, component_proportion);

	return { component_proportion, mu, sigma };
}