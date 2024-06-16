#ifndef GAT_UNIT_HPP
#define GAT_UNIT_HPP
#pragma once



#include <iostream>
#include <eigen3/Eigen/Dense>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>

using namespace std;
namespace cereal
{
  template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m)
    {
      int32_t rows = m.rows();
      int32_t cols = m.cols();
      ar(rows);
      ar(cols);
      ar(binary_data(m.data(), rows * cols * sizeof(_Scalar)));
    }

  template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols> inline
    typename std::enable_if<traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive & ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m)
    {
      int32_t rows;
      int32_t cols;
      ar(rows);
      ar(cols);

      m.resize(rows, cols);

      ar(binary_data(m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
    }
}

class GatUnit {
public:
    GatUnit() = default; 
    GatUnit(int in_features, int out_features ,double dropout, double alpha,bool concat,string name,int nhead,bool verbose);
    Eigen::MatrixXd initialize_weights(int input_size, int output_size);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj,bool isTrain);
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> backward2(const Eigen::MatrixXd& d_h_prime, const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj);
    Eigen::MatrixXd update_parameters(const Eigen::MatrixXd& param , const Eigen::MatrixXd& grad,double lr,double beta1 ,double beta2 ,double epsilon);
    Eigen::MatrixXd backward_update_parameters(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output,double lr,double beta1 ,double beta2 ,double epsilon);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> backward(const Eigen::MatrixXd& X,const Eigen::MatrixXd& d_h_prime);
    
    
private:
    int in_features;
    int out_features;
    int num_of_heads;
    double dropout;
    double alpha;
    bool concat;
    int nhead;
    Eigen::MatrixXd W;
    Eigen::MatrixXd Wh;
    Eigen::MatrixXd e;
    Eigen::MatrixXd a;
    Eigen::MatrixXd attention;
    Eigen::MatrixXd scoring_fn_target;
    Eigen::MatrixXd scoring_fn_source;
    bool verbose=false;
    std::string name;
    


    void initializeParameters();
    double dropNode(double x);

    friend class cereal::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar(in_features);
        ar(out_features);
        ar(num_of_heads);
        ar(dropout);
        ar(alpha);
        ar(verbose);
        ar(name);
        ar(nhead);
        ar(W);
        ar(a);
        ar(scoring_fn_target);
        ar(scoring_fn_source);
    }

    // Deserialization function
    template <class Archive>
    void deserialize(Archive& ar) {
        ar(in_features);
        ar(out_features);
        ar(num_of_heads);
        ar(dropout);
        ar(alpha);
        ar(verbose);
        ar(name);
        ar(nhead);
        ar(W);
        ar(a);
        ar(scoring_fn_target);
        ar(scoring_fn_source);
    }

};

#endif // GAT_UNIT_HPP
