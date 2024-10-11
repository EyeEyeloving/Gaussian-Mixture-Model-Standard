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

// ���뺯��ʵ��
Eigen::RowVectorXd KMeansSegmentation::distfun(const Eigen::MatrixXd& data_block,
	const Eigen::VectorXd& Centroids, const Eigen::VectorXd& var) {
	Eigen::MatrixXd Xmu = (data_block.colwise() - Centroids);  // X��ÿ�м�ȥC
	Eigen::MatrixXd scaled_Xmu = Xmu.array().colwise() / var.array().sqrt();  // ÿ�г��Ա�׼��sqrt(var)
	Eigen::RowVectorXd norm2_dist = scaled_Xmu.array().square().colwise().sum();  // ��ÿ��ƽ�������
	return norm2_dist;
}

void KMeansSegmentation::fit(Eigen::MatrixXd data_block) {

	// ������ȵĻ�ϱ���
	component_proportion.resize(number_component, 1.0 / number_component);

	// ���ó�ʼЭ�������
	int denom = data_block.cols();
	Eigen::VectorXd mmean = data_block.rowwise().mean();
	Eigen::VectorXd init_var = (data_block.array().colwise() - mmean.array()).array().square().rowwise().sum();
	init_var /= denom;
	// init_var += 1e-10; // ����
	Eigen::MatrixXd init_covariance = init_var.asDiagonal(); // ����Խ�Э�������
	covariance = init_covariance;
	std::cout << "Initial Covariance:\n" << covariance << std::endl;

	// �̶��������
	std::mt19937 gen(42); // ʹ�ù̶�����42��ʼ��Mersenne Twister 19937 ������
	std::uniform_int_distribution<int> dis(0, data_block.cols() - 1);  // ����������ķֲ�
	// (0, data_block.cols() - 1)��������������ɷ�Χ��Ҫ�������������Դ�0��ʼ

	Eigen::VectorXd index(number_component); // VectorXi?
	index.setZero();
	Eigen::MatrixXd Centroids(data_block.rows(), number_component); // ���ڴ洢�����ȡ����(����)
	// CentroidsͬʱҲ��ѡȡ������
	
	// ѡ���һ�����ӣ��������һ�У�
	index(0) = dis(gen); // �������һ��������
	Centroids.col(0) = data_block.col(index(0));

	// ��ʼ����С�ľ�������
	Eigen::RowVectorXd min_distance(data_block.cols());
	min_distance.setConstant(INF);

	// ��������ͨ������ģ��ѡ��
	for (int mm = 1; mm < number_component; mm++) {
		min_distance = min_distance.cwiseMin(distfun(data_block, Centroids.col(mm - 1), init_var));

		// ѡ����һ������

		double denominator = min_distance.sum();

		//if (denominator == 0 || denominator == INF) {
		//	// ���Żص�ѡ��ʣ���������
		//	std::uniform_int_distribution<int> dis_rem(0, data_block.cols() - 1);
		//}	

		// KMeans++ �� Kmeans �㷨��ѡȡ��һ������ʱ��ͬ��
		// KMeans++ ����ʣ�����ݵ㵽�����ѡ���ĵľ��룬�Ը��ʷֲ�ѡȡ�������ģ�
		// �����Զ�����ݵ���п��ܱ�ѡΪ���ġ�
		// ���ַ�ʽ���Ա����ʼ����ѡ��̫�����л�ƫ�����ݷֲ���
		
		// ����������ʣ�ʹ�þ�����Զ�ĵ�Ĳ����������
		Eigen::RowVectorXd sample_probability = min_distance / denominator;
		// ���ڲ������ʴ�����ɢ�ֲ�����
		std::discrete_distribution<> weighted_dis(sample_probability.data(), sample_probability.data() + sample_probability.size());
		
		int selected_random;
		do { // ���Żص�ѡ��ȷ���ҵ��ĵ㲻��֮ǰ���ҵ��ĵ㣨��Ϊ���ҵ��ĵ�Ҳ�����Ǹ��п��ܲ������ĵ㣩
			selected_random = weighted_dis(gen);
		} while (std::find(index.begin(), index.end(), selected_random) != index.end());
		index(mm) = selected_random;
		Centroids.col(mm) = data_block.col(index(mm));  // ��ֵ
	}

	mu = Centroids;
}