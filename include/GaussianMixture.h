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
    /*������Ϣ*/
    int number_data_dimension; // �������ݵ�ά��
    int number_data; // �������ݵ���������

    /*ģ����Ϣ*/
    int number_components; // ��˹��ֵ���Ŀ
    std::vector<double> component_proportion;
    Eigen::MatrixXd mu; // [D][K]
    Eigen::MatrixXd sigma; // [D][D] �ݲ�����[D][D][K]

    /*ģ��ѵ��*/
    bool early_stop;
    int max_iter;
    
public:
    GaussianMixture();

    GaussianMixture(int& n_Comps);

    gmModel trainGaussianMixture(Eigen::MatrixXd& data_block);

private:
    Eigen::MatrixXd validateDataInput(Eigen::MatrixXd& data_block);
};

