#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <unordered_map>
#include <vector>

Eigen::MatrixXd encode_onehot(const Eigen::VectorXd& labels);
Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits, int axis);
double accuracy(const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels, const Eigen::VectorXd& mask);
Eigen::MatrixXd logSoftmax(const Eigen::MatrixXd& input,int axis);
double convert2OneHot(double x);
Eigen::MatrixXd convertMat2OneHot(const Eigen::MatrixXd& labels);
double loss(const Eigen::MatrixXd& output,const Eigen::VectorXd& mask);
Eigen::MatrixXd nllLoss(const Eigen::MatrixXd& SoftmaxOutput, const Eigen::MatrixXd& target);
#endif