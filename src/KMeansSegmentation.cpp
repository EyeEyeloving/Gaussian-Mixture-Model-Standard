#include "KMeansSegmentation.h"

#include <random>
#include <limits>

#define INF std::numeric_limits<double>::infinity()

KMeansSegmentation::KMeansSegmentation()
	: number_component(5), share_covariance("ShareCovariance") {

}

KMeansSegmentation::KMeansSegmentation(int number_component)
	: number_component(number_component), share_covariance("ShareCovariance") {
	
}

KMeansPPlus KMeansSegmentation::getInitialParameter() const {
	return { component_proportion , mu, covariance };
}

// 距离函数实现
Eigen::RowVectorXd KMeansSegmentation::distfun(const Eigen::MatrixXd& data_block,
	const Eigen::VectorXd& Centroids, const Eigen::VectorXd& var) {
	Eigen::MatrixXd Xmu = (data_block.colwise() - Centroids);  // X中每列减去C
	Eigen::MatrixXd scaled_Xmu = Xmu.array().colwise() / var.array().sqrt();  // 每列除以标准差sqrt(var)
	Eigen::RowVectorXd norm2_dist = scaled_Xmu.array().square().colwise().sum();  // 对每列平方并求和
	return norm2_dist;
}

void KMeansSegmentation::fit(Eigen::MatrixXd data_block) {

	// 配置相等的混合比例
	component_proportion.resize(number_component, 1.0 / number_component);

	// 配置初始协方差矩阵
	int denom = data_block.cols();
	Eigen::VectorXd mmean = data_block.rowwise().mean();
	Eigen::VectorXd init_var = (data_block.array().colwise() - mmean.array()).array().square().rowwise().sum();
	init_var /= denom;
	// init_var += 1e-10; // 正则化
	Eigen::MatrixXd init_covariance = init_var.asDiagonal(); // 构造对角协方差矩阵
	covariance = init_covariance;
	std::cout << "Initial Covariance:\n" << covariance << std::endl;

	// 固定随机种子
	std::mt19937 gen(42); // 使用固定种子42初始化Mersenne Twister 19937 生成器
	std::uniform_int_distribution<int> dis(0, data_block.cols() - 1);  // 随机列索引的分布
	// (0, data_block.cols() - 1)是随机整数的生成范围：要生成索引，所以从0开始

	Eigen::VectorXd index(number_component); // VectorXi?
	index.setZero();
	Eigen::MatrixXd Centroids(data_block.rows(), number_component); // 用于存储随机抽取的列(样本)
	// Centroids同时也是选取的质心
	
	// 选择第一个种子（随机采样一列）
	index(0) = dis(gen); // 随机生成一个列索引
	Centroids.col(0) = data_block.col(index(0));

	// 初始化最小的距离向量
	Eigen::RowVectorXd min_distance(data_block.cols());
	min_distance.setConstant(INF);

	// 后续种子通过概率模型选择
	for (int mm = 1; mm < number_component; mm++) {
		min_distance = min_distance.cwiseMin(distfun(data_block, Centroids.col(mm - 1), init_var));

		// 选择下一个种子

		double denominator = min_distance.sum();

		//if (denominator == 0 || denominator == INF) {
		//	// 不放回的选择剩余随机种子
		//	std::uniform_int_distribution<int> dis_rem(0, data_block.cols() - 1);
		//}	

		// KMeans++ 与 Kmeans 算法在选取下一个质心时不同：
		// KMeans++ 根据剩余数据点到最近已选质心的距离，以概率分布选取后续质心；
		// 距离较远的数据点更有可能被选为质心。
		// 这种方式可以避免初始质心选得太过集中或偏离数据分布。
		
		// 计算采样概率：使得距离最远的点的采样概率最高
		Eigen::RowVectorXd sample_probability = min_distance / denominator;
		// 基于采样概率创建离散分布对象
		std::discrete_distribution<> weighted_dis(sample_probability.data(), sample_probability.data() + sample_probability.size());
		
		int selected_random;
		do { // 不放回的选择；确保找到的点不是之前已找到的点（因为已找到的点也可能是更有可能采样到的点）
			selected_random = weighted_dis(gen);
		} while (std::find(index.begin(), index.end(), selected_random) != index.end());
		index(mm) = selected_random;
		Centroids.col(mm) = data_block.col(index(mm));  // 赋值
	}

	mu = Centroids;
}