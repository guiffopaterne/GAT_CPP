#ifndef GAT_LAYER_HPP
#define GAT_LAYER_HPP
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/access.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <tuple>
#include "GatUnit.hpp"

using namespace std;

class GatLayer {
public:
    GatLayer() = default; 
    GatLayer(int in_features, int out_features,int num_of_heads ,double dropout, double alpha, bool concat,string name,bool verbose);
    Eigen::MatrixXd stack_heads(const std::vector<Eigen::MatrixXd>& head_outputs);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj,bool isTrain);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& adj, const Eigen::MatrixXd& gradput,double lr,double beta1 ,double beta2 ,double epsilon);
    Eigen::MatrixXd sum_heads(const std::vector<Eigen::MatrixXd>& head_outputs) ;

private:
    int in_features;
    int out_features;
    int num_of_heads;
    double dropout;
    double alpha;
    Eigen::MatrixXd X;
    std::vector<GatUnit> layer;
    bool concat=true;
    bool verbose=false;
    string name;

    // friend class boost::serialization::access;

    // template <typename Archive>
    // void GatLayer::serialize(Archive& ar, const unsigned int version) ;

};

#endif // GAT_LAYER_HPP