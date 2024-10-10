#pragma once

#include "../include/Eigen/Dense"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

class ExpectationMaximization
{
public:
	int number_data_dimension; // �������ݵ�ά��
	double epsilon_early_stop;
	bool early_stop;
	int max_iter;
	double prob_cutoff;
	
public:
	ExpectationMaximization();

	ExpectationMaximization(int n_Dims);
	
	void trainExpectationMaximization(const Eigen::MatrixXd data_block, const int& number_components, 
		Eigen::MatrixXd mu, Eigen::MatrixXd sigma, std::vector<double> component_proportion);
	
private:
	/**
	 * ������������ÿ�����ݵ㣨����D��ά�ȣ��������ȣ�E-step: ����������
	 *
	 * @param data_stream ��������������ʾ�۲����ݵļ��ϡ�
	 *        ����: Eigen::MatrixXd
	 *        ά��: [D]������ D Ϊ���ݵ��ά�ȵ�������
	 *		  ����: ����淶Ϊ��������
	 *
	 * @param mixing_proportion ��ϱ�������ʾÿ���ɷ��ڻ�Ϸֲ��е�Ȩ�ء�
	 *        ����: double
	 *        ά��: [1]������������ֵ����
	 *
	 * @param mu ��ֵ��������ʾ��˹�ֲ���ÿ���ɷֵľ�ֵ��
	 *        ����: Eigen::MatrixXd
	 *        ά��: [D][K]������ D Ϊ�����ĳ����˹��ֵ�ά�ȣ�K ����ֵ���Ŀ
	 *
	 * @param sigma Э������󣬱�ʾ��˹�ֲ���ÿ���ɷֵı�׼�
	 *        ����: Eigen::MatrixXd
	 *        ά��: [D][D]������ D Ϊ�����ĳ����˹��ֵ�ά�ȡ�
	 *
	 * @return double ������������ÿ�����ݵ��������ֵ��
	 *         ����: double
	 *         ά��: [1]�������������������һ�¡�
	 */
	Eigen::RowVectorXd estimateExpectationStep(const Eigen::MatrixXd data_block, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sigma);
    // ���������ظ������ ����&

	double updateComponentProportion(const Eigen::RowVectorXd& responsibility);

	Eigen::VectorXd updateMu(Eigen::MatrixXd data_block, const Eigen::RowVectorXd& responsibility);

	Eigen::MatrixXd updateSigma(const Eigen::MatrixXd data_block, const Eigen::VectorXd& mu, const Eigen::RowVectorXd& responsibility);

};

