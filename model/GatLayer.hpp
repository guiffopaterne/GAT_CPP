#ifndef GAT_LAYER_HPP
#define GAT_LAYER_HPP
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
    std::string name;

    friend class cereal::access;
     template <class Archive>
    void deserialize(Archive& ar, unsigned int version) {
        if (version > 1) {
        cout<<"Premiere version"<<endl;
        }
        ar(in_features);
        ar(out_features);
        ar(num_of_heads);
        ar(dropout);
        ar(alpha);
        ar(layer); 
        ar(concat);
        ar(verbose);
        ar(name);
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned int version){
        if (version > 1) {
        cout<<"Premiere version"<<endl;
        }
        ar(in_features);
        ar(out_features);
        ar(num_of_heads);
        ar(dropout);
        ar(alpha);
        ar(layer); 
        ar(concat);
        ar(verbose);
        ar(name);
        }

};

#endif // GAT_LAYER_HPP