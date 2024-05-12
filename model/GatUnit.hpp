#ifndef GAT_UNIT_HPP
#define GAT_UNIT_HPP
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/access.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <tuple>

using namespace std;

class GatUnit {
public:
    GatUnit() = default; 
    GatUnit(int in_features, int out_features ,double dropout, double alpha,string name,int nhead,bool verbose);
    Eigen::MatrixXd initialize_weights(int input_size, int output_size);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj,bool isTrain);
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output );
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> backward2(const Eigen::MatrixXd& d_h_prime, const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj);
    Eigen::MatrixXd update_parameters(const Eigen::MatrixXd& param , const Eigen::MatrixXd& grad,double lr,double beta1 ,double beta2 ,double epsilon);
    Eigen::MatrixXd backward_update_parameters(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output,double lr,double beta1 ,double beta2 ,double epsilon);
private:
    int in_features;
    int out_features;
    int num_of_heads;
    double dropout;
    double alpha;
    Eigen::MatrixXd W;
    Eigen::MatrixXd Wh;
    Eigen::MatrixXd e;
    Eigen::MatrixXd a;
    Eigen::MatrixXd attention;
    Eigen::MatrixXd scoring_fn_target;
    Eigen::MatrixXd scoring_fn_source;
    bool verbose=false;
    string name;
    int nhead;


    void initializeParameters();
    double dropNode(double x);

    // friend class boost::serialization::access;

    // template <typename Archive>
    // void GatUnit::serialize(Archive& ar, const unsigned int version) ;

};

#endif // GAT_UNIT_HPP
