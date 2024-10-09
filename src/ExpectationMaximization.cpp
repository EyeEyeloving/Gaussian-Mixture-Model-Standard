#define ppi 3.1415926535354

#include "ExpectationMaximization.h"

ExpectationMaximization::ExpectationMaximization()
	: number_data_dimension(1), epsilon_early_stop(1e-3), early_stop(false), max_iter(100) {

}

ExpectationMaximization::ExpectationMaximization(int n_Dims)
	: number_data_dimension(n_Dims), epsilon_early_stop(1e-3), early_stop(false), max_iter(100) {

}

void ExpectationMaximization::trainExpectationMaximization(const Eigen::MatrixXd data_block, const int& number_components, 
	Eigen::MatrixXd mu, Eigen::MatrixXd sigma, std::vector<double> component_proportion) {

	/*数据信息*/
	// Eigen::Index number_data = data_block.cols();
	int number_data = data_block.cols();
	// double number_data = data_block.cols();
	// 返回的是 Eigen::Index 类型，通常它是 ptrdiff_t 类型，和 double 并不一致。
	// int number_components = component_proportion.size();

	/*EM迭代*/
	/*定义中间参数*/
	Eigen::MatrixXd responsibility(number_components, number_data);
	responsibility.setZero();
	// std::vector<std::vector<double>> responsibility(number_data, std::vector<double>(number_components, 0.0)); // 响应度；二维数组初始化
	double logliklihood_prev = 0;
	int EM_step = 0;
	double epsilon_cur = 1;

	while (!early_stop && EM_step < max_iter) {
		++EM_step;
		std::cout << "The current EM-step is: " << EM_step << std::endl;

		/*E-step*/
		Eigen::VectorXd response_sum = Eigen::VectorXd::Zero(number_data);
		//Eigen::VectorXd response_sum(number_data, 0.0);
		for (int j = 0; j < number_components; j++) {
			Eigen::VectorXd mu_col = mu.col(j);
			responsibility.col(j) = component_proportion[j] * estimateExpectationStep(data_block, mu_col, sigma);
			response_sum = response_sum + responsibility.col(j); // response_sum += responsibility[j];
		}
		for (int j = 0; j < number_components; j++) {
			responsibility.col(j).array() /= response_sum.array();
		}

		///*M-step*/
		//for (int j = 0; j < number_components; j++) {
		//	const Eigen::RowVectorXd& response_jrow = responsibility.row(j); // 非常量引用必须为左值
		//	component_proportion[j] = 1 / number_data * updateComponentProportion(response_jrow);
		//	mu.col(j) = 1 / response_jrow.sum() * updateMu(data_block, response_jrow);
		//	sigma += 1 / number_data * updateSigma(data_block, mu.col(j), response_jrow);
		//}

		///*check convergency*/
		//double logliklihood_cur = updatedNegLogLikLiHood(data_block, number_components, mu, sigma, component_proportion);
		//epsilon_cur = (logliklihood_cur - logliklihood_prev) / logliklihood_prev;
		//std::cout << epsilon_cur << std::endl;
		//if (epsilon_cur < epsilon_early_stop) early_stop = true;
	}
}

Eigen::VectorXd ExpectationMaximization::estimateExpectationStep(const Eigen::MatrixXd data_block, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma) {
	double magnitude = std::pow(std::pow(2 * ppi, 2) * sigma.determinant(), -0.5);
	Eigen::MatrixXd Xmu = data_block.colwise() - mu;

	if (sigma.determinant() == 0) std::cerr << "The Covariance of GMM is not Invertible" << std::endl;
	// Eigen::MatrixXd XX = Xmu.transpose() * sigma.inverse().array().sqrt(); // array()会改变数据结构
	Eigen::LLT<Eigen::MatrixXd> llt(sigma.inverse());
	if (llt.info() != Eigen::Success) std::cerr << "The Covariance of GMM is not Positive Definite" << std::endl;
	// Eigen::MatrixXd sigma_inverse_sqrt = llt.matrixL() * llt.matrixL().transpose();
	// matrixL() 返回的是一个 TriangularView 对象，这个对象不能直接进行乘法
	Eigen::MatrixXd XX = Xmu.transpose() * (llt.matrixL().toDenseMatrix() * llt.matrixL().toDenseMatrix().transpose());
	Eigen::MatrixXd exponent = -0.5 * XX.array().square().rowwise().sum();
	return magnitude * exponent.array().exp().transpose();// 返回行向量
}

double ExpectationMaximization::updateComponentProportion(const Eigen::RowVectorXd& responsibility) {
	return responsibility.sum(); // 还需要除以数据的大小
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