#ifndef GAT_LAYER_HPP
#define GAT_LAYER_HPP
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/access.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <tuple>

using namespace std;

class GatLayer {
public:
    GatLayer() = default; 
    GatLayer(int in_features, int out_features, double dropout, double alpha, bool concat);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj) ;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& gradput);

    // std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> backward2(const Eigen::MatrixXd& h, const Eigen::MatrixXd& gradput);
    Eigen::MatrixXd update_parameters(const Eigen::MatrixXd& param, const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient);
    void backward_update_parameters(const Eigen::MatrixXd& h, const Eigen::MatrixXd& grad_h);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& gradput);

private:
    int in_features;
    int out_features;
    double dropout;
    double alpha;
    bool concat;
    Eigen::MatrixXd W;
    Eigen::MatrixXd a;

    void initializeParameters();

    Eigen::MatrixXd prepareAttentionalMechanismInput(const Eigen::MatrixXd& Wh);

    double leakyReLU(double x);
    double dropNode(double x);

    
    Eigen::MatrixXd prepareAttentionalMechanismInputPrime(const Eigen::MatrixXd& Wh) ;

    double leakyReLUPrime(double x) ;

    // Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits, int axis) ;

    Eigen::MatrixXd attention_backward(const Eigen::MatrixXd& grad_h);

    Eigen::MatrixXd attention_derivative();

    double elu(double x);

    // friend class boost::serialization::access;

    // template <typename Archive>
    // void GatLayer::serialize(Archive& ar, const unsigned int version) ;

};

#endif // GAT_LAYER_HPP