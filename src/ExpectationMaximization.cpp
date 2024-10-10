#define ppi 3.1415926535354
#define custsys_realmin 1e-10

#include "ExpectationMaximization.h"

ExpectationMaximization::ExpectationMaximization()
	: number_data_dimension(1), epsilon_early_stop(1e-3), early_stop(false), max_iter(100), prob_cutoff(1e-8) {

}

ExpectationMaximization::ExpectationMaximization(int n_Dims)
	: number_data_dimension(n_Dims), epsilon_early_stop(1e-3), early_stop(false), max_iter(100), prob_cutoff(1e-8) {

}

void ExpectationMaximization::trainExpectationMaximization(const Eigen::MatrixXd data_block, const int& number_components, 
	Eigen::MatrixXd mu, Eigen::MatrixXd sigma, std::vector<double> component_proportion) {

	/*������Ϣ*/

	int number_data = data_block.cols();
	number_data_dimension = data_block.rows();

	// Eigen::Index number_data = data_block.cols();
	// double number_data = data_block.cols();
	// �ɴ��룺data_block.cols()���ص��� Eigen::Index ���ͣ�ͨ������ ptrdiff_t ���ͣ��� double ����һ�¡�

	/*EM����*/

	// �����м��������Ӧ�ȣ������ȣ�
	Eigen::MatrixXd responsibility(number_components, number_data);
	responsibility.setZero();

	// std::vector<std::vector<double>> responsibility(number_data, std::vector<double>(number_components, 0.0)); 
	// �ɴ��룺��ά�����ʼ��

	double logliklihood_prev = 0;
	int EM_step = 0;
	double epsilon_cur = 1;

	while (!early_stop && EM_step < max_iter) {

		// ��ӡ��ǰ��EM��������
		++EM_step;
		std::cout << "The current EM-step is: " << EM_step << std::endl;

		/*E-step: ����ÿһ�����֮���������ݵ�������*/

		Eigen::RowVectorXd response_sum(number_data);
		response_sum.setZero();

		// std::cout << INT_MIN << std::endl;
		// �ɴ��룺INT_MIN��ֵ��-2147483648������ʵ��������realmin���ܣ�Ӧ>0��

		// ����������
		for (int j = 0; j < number_components; j++) {
			Eigen::VectorXd mu_col = mu.col(j);
			responsibility.row(j) = component_proportion[j] * estimateExpectationStep(data_block, mu_col, sigma);
			
			// response_sum += responsibility.row(j);
		}
		response_sum = responsibility.colwise().sum();

		// ��һ������ responsibility �����ÿһ�н��й�һ��
		responsibility.array().rowwise() /= (response_sum.array() + custsys_realmin);
		// �ضϣ���С�� prob_cutoff ��Ԫ����Ϊ 0
		responsibility = (responsibility.array() >= prob_cutoff).select(responsibility, 0);
		// ���¹�һ������к�
		response_sum = responsibility.colwise().sum();
		// �ٴι�һ������ responsibility �����ÿһ�н��й�һ��
		responsibility.array().rowwise() /= (response_sum.array() + custsys_realmin);

		std::cout << responsibility.transpose() << std::endl;

		//// ÿһ���������й�һ��
		//for (int j = 0; j < number_components; j++) {
		//	// �ȶ�ÿһ��������һ��
		//	responsibility.row(j).array() /= (response_sum.array() + custsys_realmin);
		//	
		//	// �ض�
		//	responsibility.row(j) = (responsibility.row(j).array() >= prob_cutoff).select(responsibility.row(j), 0);
		//	// ����һ���� responsibility.row(j) ��ͬ��С�Ĳ��� Eigen::Array
		//	// mask.select(A, B)�����ǣ����� mask �е�ÿ��Ԫ�أ������Ӧλ��Ϊ true����ѡ�� A �еĶ�ӦԪ�أ������Ӧλ��Ϊ false����ѡ�� B �еĶ�ӦԪ�ء�
		//}
		//response_sum = responsibility.colwise().sum();
		//for (int j = 0; j < number_components; j++) {
		//	responsibility.row(j).array() /= (response_sum.array() + custsys_realmin);
		//}
		//�ɴ��룺forѭ���ܶ࣬��Ҫ����

		///*M-step*/
		//for (int j = 0; j < number_components; j++) {
		//	const Eigen::RowVectorXd& response_jrow = responsibility.row(j); // �ǳ������ñ���Ϊ��ֵ
		//	component_proportion[j] = 1 / number_data * updateComponentProportion(response_jrow);
		//	mu.col(j) = 1 / response_jrow.sum() * updateMu(data_block, response_jrow);
		//	sigma += 1 / number_data * updateSigma(data_block, mu.col(j), response_jrow);
		//}

		///*check convergency*/
		//double logliklihood_cur = updatedNegLogLikLiHood(data_block, number_components, mu, sigma, component_proportion);
		//epsilon_cur = (logliklihood_cur - logliklihood_prev) / logliklihood_prev;
		//std::cout << epsilon_cur << std::endl;
		//if (epsilon_cur < epsilon_early_stop) early_stop = true;

		std::cin.get();
	}
}

Eigen::RowVectorXd ExpectationMaximization::estimateExpectationStep(const Eigen::MatrixXd data_block, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma) {
	// ����ϵ������ܷ�ĸΪ0�����
	double magnitude = std::pow(2 * ppi, -0.5 * number_data_dimension) * std::pow(sigma.determinant() + custsys_realmin, -0.5); 
	Eigen::MatrixXd Xmu = data_block.colwise() - mu;

	if (sigma.determinant() == 0) std::cerr << "The Covariance of GMM is not Invertible" << std::endl;

	// Eigen::MatrixXd XX = Xmu.transpose() * sigma.inverse().array().sqrt();
	// �ɴ��룺array()��ı����ݽṹ������ֱ������ʹ��

	// ʹ��LLT�ֽ⴦��Э�������
	Eigen::LLT<Eigen::MatrixXd> llt(sigma);
	if (llt.info() != Eigen::Success) std::cerr << "The Covariance of GMM is not Positive Definite" << std::endl;

	// Eigen::MatrixXd sigma_inverse_sqrt = llt.matrixL() * llt.matrixL().transpose();
	// �ɴ��룺matrixL() ���ص���һ�� TriangularView �������������ֱ�ӽ��г˷�

	Eigen::MatrixXd L = llt.matrixL().toDenseMatrix();
	Eigen::MatrixXd L_inv = L.inverse(); // ���� L �������

	// Eigen::MatrixXd L_inv = llt.matrixL().toDenseMatrix() * llt.matrixL().toDenseMatrix().transpose();
	// �ɴ��룺���������������Э�����������ƽ��
	// std::cout << "The Inverse() of Sigma: \n" << L_inv << std::endl;

	Eigen::MatrixXd XX = L_inv * Xmu; // [D][N]

	// std::cout << Xmu.coeff(0, 0) << " " << Xmu.coeff(1, 0) << " " << Xmu.coeff(2, 0) << std::endl;
	// ���Դ��룺���Xmu�Ĳ���ֵ
	// std::cout << XX.coeff(0, 0) << " " << XX.coeff(1, 0) << " " << XX.coeff(2, 0) << std::endl;
	// ���Դ��룺���XX�Ĳ���ֵ

	Eigen::MatrixXd exponent = -0.5 * XX.array().square().colwise().sum();
	return magnitude * exponent.array().exp();// ����������
}

double ExpectationMaximization::updateComponentProportion(const Eigen::RowVectorXd& responsibility) {
	return responsibility.sum(); // ����Ҫ�������ݵĴ�С
}

Eigen::VectorXd ExpectationMaximization::updateMu(Eigen::MatrixXd data_block, const Eigen::RowVectorXd& responsibility) {
	return (data_block.array().rowwise() * responsibility.array()).rowwise().sum(); 
}

//Eigen::MatrixXd ExpectationMaximization::updateSigma(const Eigen::MatrixXd data_block, const Eigen::VectorXd& mu, const Eigen::RowVectorXd& responsibility) {
//	Eigen::MatrixXd Xmu = (data_block.colwise() - mu).rowwise()*responsibility.array().sqrt();
//	return Xmu * Xmu.transpose();
//}

//double ExpectationMaximization::updatedNegLogLikLiHood(const Eigen::MatrixXd data_block, 
//	const int& number_components, const Eigen::MatrixXd& mu, Eigen::MatrixXd& sigma, std::vector<double>& component_proportion) {
//	Eigen::VectorXd response_sum(data_block.cols(), 0.0);
//	for (int j = 0; j < number_components; j++) {
//		const Eigen::VectorXd& mu_col = mu.col(j);
//		response_sum += component_proportion[j] * estimateExpectationStep(data_block, mu_col, sigma);
//	}
//	return -response_sum.array().log().mean();
//}