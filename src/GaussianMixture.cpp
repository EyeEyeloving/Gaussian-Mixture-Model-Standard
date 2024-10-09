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
	if (data_block.rows() > data_block.cols()) return data_block.transpose(); // ����������ת��Ϊ[D][N]
	// ԭ����data_block = data_block.transpose()����data_blockת�ú�ά�����������������

	/*����������/ָ��Ϳ�����void���*/
}

gmModel GaussianMixture::trainGaussianMixture(Eigen::MatrixXd& data_block_raw) {

	/*������������*/
	Eigen::MatrixXd data_block = validateDataInput(data_block_raw);
	number_data_dimension = data_block.rows(); // ���ݵ�ά��
	number_data = data_block.cols(); // ������

	/*����GMM����*/
	component_proportion.resize(number_components, 1/number_components);
	mu.resize(number_data_dimension, number_components);
	sigma.resize(number_data_dimension, number_data_dimension);
	
	/*����KMeans++�����ĳ�ʼ��*/
	KMeansPP kmeanspp(number_components, 100);
	kmeanspp.fit(data_block);
	auto output = kmeanspp.getMeansAndCovariances(data_block);
	mu = output.first;
	auto covariances = output.second;
	for (int j = 0; j < number_components; j++) {
		sigma += (1 / number_components * covariances.block(0, j * data_block.rows(), data_block.rows(), data_block.rows()));
	}

	/* ��ӡ mu �� sigma ��ֵ */
	std::cout << "Mu (Means):\n" << mu << std::endl;
	std::cout << "Sigma (Covariances):\n" << sigma << std::endl;

	/*ѵ��*/
	trainExpectationMaximization(data_block, number_components, mu, sigma, component_proportion);

	return { component_proportion, mu, sigma };
}