#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <unordered_map>
#include <cereal/cereal.hpp>
#include <vector>
using namespace std;

Eigen::MatrixXd encode_onehot(const Eigen::VectorXd& labels,int num_class);
Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits, int axis);
double accuracy(const Eigen::VectorXd& output, const Eigen::VectorXd& labels, const Eigen::VectorXi& mask);
Eigen::MatrixXd logSoftmax(const Eigen::MatrixXd& input,int axis);
Eigen::MatrixXd logSoftmax(const Eigen::MatrixXd& input);
double convert2OneHot(double x);
Eigen::MatrixXd convertMat2OneHot(const Eigen::MatrixXd& labels);
double loss(const Eigen::MatrixXd& output,const Eigen::VectorXi& mask);
double nllLoss(const Eigen::MatrixXd& SoftmaxOutput, const Eigen::MatrixXd& target,const Eigen::VectorXi& mask);
double nllLoss(const Eigen::MatrixXd& Output, const Eigen::MatrixXd& target,const Eigen::VectorXi& mask,int axis);
std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi> createMask(const int size,const int train_size, const int val_size, bool isShuffle);
double elu(double x,double alpha);
double leakyReLU(double x,double alpha);
double leakyReLUPrime(double x,double alpha);
double dropNode(double x,double dropout) ;
Eigen::MatrixXd softmax_prime(const Eigen::MatrixXd& S);
double dropNode_prime(double x, double dropout) ;
void shape(const Eigen::MatrixXd& m,const string name,bool verbose);

Eigen::VectorXd argmax(Eigen::MatrixXd m);
int argmax_vector(const Eigen::VectorXd& v);
Eigen::MatrixXd encode_onehot_out(const Eigen::MatrixXd& output);
Eigen::MatrixXd encode_onehot(const Eigen::VectorXd& labels,int num_class);
std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::VectorXd> create_sub_data(const Eigen::MatrixXd& features_matrix ,const Eigen::MatrixXd& adj_matrix,const Eigen::VectorXd& labels_vector, const Eigen::VectorXi& mask);
Eigen::VectorXd create_sub_labels_vector(const Eigen::VectorXd& labels_vector, const Eigen::VectorXi& mask);
Eigen::MatrixXd create_sub_adjacency_matrix(const Eigen::MatrixXd& adj_matrix, const Eigen::VectorXi& mask);

// sans mask
double loss(const Eigen::MatrixXd& output);
double nllLoss(const Eigen::MatrixXd& SoftmaxOutput, const Eigen::MatrixXd& target);
double nllLoss(const Eigen::MatrixXd& Output, const Eigen::MatrixXd& target,int axis);
double accuracy(const Eigen::VectorXd& output, const Eigen::VectorXd& labels);



void head(const Eigen::MatrixXd& M,string name,int ligne);


#endif