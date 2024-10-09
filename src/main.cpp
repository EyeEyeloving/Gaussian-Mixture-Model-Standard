#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "../include/Eigen/Dense"
#include "GaussianMixture.h"

int main() {
	const std::string filename = "dat/data_trajectory_1.txt";
	std::ifstream input_file(filename);

	std::vector<std::vector<double>> filedata_array;
	Eigen::MatrixXd data_block;
	double num1, num2, num3;

	while (input_file >> num1 >> num2 >> num3) {
		filedata_array.emplace_back(std::vector<double>{num1, num2, num3});
	}
	input_file.close();

	data_block.resize(filedata_array.size(), 3);
	for (int m = 0; m < filedata_array.size(); m++) {
		data_block.row(m) = Eigen::VectorXd::Map(filedata_array[m].data(), filedata_array[m].size()); // ¶ÔÏóÊÓÍ¼£¿
	}

	/*GMM*/
	int number_components = 5;
	GaussianMixture gmModel(number_components);
	gmModel.trainGaussianMixture(data_block);
	
}