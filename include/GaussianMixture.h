#pragma once

#include "ExpectationMaximization.h"
#include <iostream>
#include <vector>
#include "../include/Eigen/Dense"

struct gmModel
{
    std::vector<double> component_proportion;
    Eigen::MatrixXd mu; // [D][K]
    Eigen::MatrixXd sigma; // [D][D]
};

class GaussianMixture :
    public ExpectationMaximization
{
public:
    /*数据信息*/
    int number_data_dimension; // 输入数据的维度
    int number_data; // 输入数据的样本数量

    /*模型信息*/
    int number_components; // 高斯组分的数目
    std::vector<double> component_proportion;
    Eigen::MatrixXd mu; // [D][K]
    Eigen::MatrixXd sigma; // [D][D] 暂不考虑[D][D][K]

    /*模型训练*/
    bool early_stop;
    int max_iter;
    
public:
    GaussianMixture();

    GaussianMixture(int& n_Comps);

    gmModel trainGaussianMixture(Eigen::MatrixXd& data_block);

private:
    Eigen::MatrixXd validateDataInput(Eigen::MatrixXd& data_block);
};

