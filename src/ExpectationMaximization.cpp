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

	/*数据信息*/

	int number_data = data_block.cols();
	number_data_dimension = data_block.rows();

	// Eigen::Index number_data = data_block.cols();
	// double number_data = data_block.cols();
	// 旧代码：data_block.cols()返回的是 Eigen::Index 类型，通常它是 ptrdiff_t 类型，和 double 并不一致。

	/*EM迭代*/

	// 定义中间参数：响应度（隶属度）
	Eigen::MatrixXd responsibility(number_components, number_data);
	responsibility.setZero();

	// std::vector<std::vector<double>> responsibility(number_data, std::vector<double>(number_components, 0.0)); 
	// 旧代码：二维数组初始化

	double neglogliklihood_prev = INT_MAX;
	int EM_step = 0;
	double epsilon_cur = 1;
	Eigen::MatrixXd ASigma(number_data_dimension, number_data_dimension);

	while (!early_stop && EM_step < max_iter) {

		/*E-step: 计算每一个组分之于输入数据的隶属度*/

		Eigen::RowVectorXd response_sum(number_data);
		response_sum.setZero();

		// std::cout << INT_MIN << std::endl;
		// 旧代码：INT_MIN的值是-2147483648，不能实现期望的realmin功能（应>0）

		// 计算隶属度
		for (int j = 0; j < number_components; j++) {
			Eigen::VectorXd mu_col = mu.col(j);
			responsibility.row(j) = component_proportion[j] * estimateExpectationStep(data_block, mu_col, sigma);
			
			// response_sum += responsibility.row(j);
		}
		response_sum = responsibility.colwise().sum();

		// 归一化：对 responsibility 矩阵的每一行进行归一化
		responsibility.array().rowwise() /= (response_sum.array() + custsys_realmin);
		// 截断：将小于 prob_cutoff 的元素置为 0
		responsibility = (responsibility.array() >= prob_cutoff).select(responsibility, 0);
		// 更新归一化后的列和
		response_sum = responsibility.colwise().sum();
		// 再次归一化：对 responsibility 矩阵的每一行进行归一化
		responsibility.array().rowwise() /= (response_sum.array() + custsys_realmin);

		// std::cout << responsibility.transpose() << std::endl;

		//// 每一个样本进行归一化
		//for (int j = 0; j < number_components; j++) {
		//	// 先对每一个样本归一化
		//	responsibility.row(j).array() /= (response_sum.array() + custsys_realmin);
		//	
		//	// 截断
		//	responsibility.row(j) = (responsibility.row(j).array() >= prob_cutoff).select(responsibility.row(j), 0);
		//	// 返回一个与 responsibility.row(j) 相同大小的布尔 Eigen::Array
		//	// mask.select(A, B)含义是：对于 mask 中的每个元素，如果对应位置为 true，则选择 A 中的对应元素；如果对应位置为 false，则选择 B 中的对应元素。
		//}
		//response_sum = responsibility.colwise().sum();
		//for (int j = 0; j < number_components; j++) {
		//	responsibility.row(j).array() /= (response_sum.array() + custsys_realmin);
		//}
		//旧代码：for循环很多，需要精简

		/*check convergency: 检查收敛性*/

		// 将std::vector<double>转换为Eigen::VectorXd
		Eigen::RowVectorXd ComponentProportions = Eigen::Map<const Eigen::RowVectorXd>(component_proportion.data(), component_proportion.size());
		Eigen::MatrixXd exp_loglik = responsibility.array().colwise() * ComponentProportions.transpose().array();
		double neg_logliklihood_cur = -(exp_loglik.rowwise().sum().array() + custsys_realmin).log().mean();
		// double logliklihood_cur = updatedNegLogLikLiHood(data_block, number_components, mu, sigma, component_proportion);

		epsilon_cur = std::abs((neg_logliklihood_cur - neglogliklihood_prev) / neglogliklihood_prev);
		std::cout << epsilon_cur << std::endl;
		if (epsilon_cur < epsilon_early_stop) {
			early_stop = true;
			break;
		}
		neglogliklihood_prev = neg_logliklihood_cur;

		/*增加EM-step*/

		// 打印当前的EM迭代步次
		++EM_step;
		std::cout << "The current EM-step is: " << EM_step << std::endl;

		/*M-step: 当还未收敛时，则更新参数*/
		ASigma.setZero();
		for (int j = 0; j < number_components; j++) {
			const Eigen::RowVectorXd& response_jrow = responsibility.row(j); // 非常量引用必须为左值
			component_proportion[j] = updateComponentProportion(response_jrow);
			mu.col(j) = updateMu(data_block, response_jrow);
			ASigma += updateSigma(data_block, mu.col(j), response_jrow);
		}
		ASigma /= number_data;
		sigma = ASigma;

		// std::cin.get();
	}
}

Eigen::RowVectorXd ExpectationMaximization::estimateExpectationStep(const Eigen::MatrixXd data_block, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma) {
	// 计算系数，规避分母为0的情况
	double magnitude = std::pow(2 * ppi, -0.5 * number_data_dimension) * std::pow(sigma.determinant() + custsys_realmin, -0.5); 
	Eigen::MatrixXd Xmu = data_block.colwise() - mu;

	if (sigma.determinant() == 0) std::cerr << "The Covariance of GMM is not Invertible" << std::endl;

	// Eigen::MatrixXd XX = Xmu.transpose() * sigma.inverse().array().sqrt();
	// 旧代码：array()会改变数据结构，不能直接这样使用

	// 使用LLT分解处理协方差矩阵
	Eigen::LLT<Eigen::MatrixXd> llt(sigma);
	if (llt.info() != Eigen::Success) std::cerr << "The Covariance of GMM is not Positive Definite" << std::endl;

	// Eigen::MatrixXd sigma_inverse_sqrt = llt.matrixL() * llt.matrixL().transpose();
	// 旧代码：matrixL() 返回的是一个 TriangularView 对象，这个对象不能直接进行乘法

	Eigen::MatrixXd L = llt.matrixL().toDenseMatrix();
	Eigen::MatrixXd L_inv = L.inverse(); // 计算 L 的逆矩阵

	// Eigen::MatrixXd L_inv = llt.matrixL().toDenseMatrix() * llt.matrixL().toDenseMatrix().transpose();
	// 旧代码：这个不是用来处理协方差矩阵的逆的平方
	// std::cout << "The Inverse() of Sigma: \n" << L_inv << std::endl;

	Eigen::MatrixXd XX = L_inv * Xmu; // [D][N]

	// std::cout << Xmu.coeff(0, 0) << " " << Xmu.coeff(1, 0) << " " << Xmu.coeff(2, 0) << std::endl;
	// 调试代码：输出Xmu的部分值
	// std::cout << XX.coeff(0, 0) << " " << XX.coeff(1, 0) << " " << XX.coeff(2, 0) << std::endl;
	// 调试代码：输出XX的部分值

	Eigen::MatrixXd exponent = -0.5 * XX.array().square().colwise().sum();
	return magnitude * exponent.array().exp();// 返回行向量
}

double ExpectationMaximization::updateComponentProportion(const Eigen::RowVectorXd& responsibility) {
	return responsibility.mean(); 
}

Eigen::VectorXd ExpectationMaximization::updateMu(Eigen::MatrixXd data_block, const Eigen::RowVectorXd& responsibility) {
	return (data_block * responsibility.transpose()) / responsibility.sum();

	// 在matlab的gmcluster_learn函数中，利用了已计算的component_proportion
}

Eigen::MatrixXd ExpectationMaximization::updateSigma(const Eigen::MatrixXd data_block, const Eigen::VectorXd& mu, const Eigen::RowVectorXd& responsibility) {
	Eigen::MatrixXd Xmu = data_block.array().colwise() - mu.array();
	Eigen::MatrixXd XX = Xmu.array().rowwise() * responsibility.array().sqrt();  // 逐元素乘法
	return XX * XX.transpose();  // 矩阵乘法
}
