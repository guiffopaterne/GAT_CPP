#ifndef GAT_NETWORK_HPP
#define GAT_NETWORK_HPP
#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <cmath>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>


#include "GatLayer.hpp"

using namespace std;


class GAT {
public:
    GAT() = default; 
    GAT(int num_of_layers,std::vector<int> nhead,std::vector<int> num_features_per_layer,int nclass, double dropout, double alpha,bool verbose);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& adj,int num_thread,int num_thread_head,bool isTrain);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& adj, int num_thread,int num_thread_head,const Eigen::MatrixXd& grad_output,double lr,double beta1 ,double beta2 ,double epsilon);
    std::string print_representation();

    // void train(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
    //             const Eigen::VectorXd& labels, const Eigen::VectorXi& train_mask, const Eigen::VectorXi& val_mask,
    //             int num_epochs,double lr,double beta1,double beta2,double epsilon,int patience, int early_stop) ;

    std::tuple<double,double> evaluate(const Eigen::MatrixXd& node_features, const Eigen::MatrixXd& adj, int num_thread,const Eigen::VectorXd& labels,const Eigen::VectorXi& val_mask) ;
    
    void set_num_epochs(int num_epochs);
    int get_num_epoch();
    int get_num_epochs();
    int get_nclass();
    int get_num_of_layers();
    std::vector<int> get_num_head_per_layer();
    std::vector<int> get_num_features_per_layer();
    std::vector<GatLayer> get_attentions();

    


private:
    friend class cereal::access;
    double dropout;
    double alpha;
    int num_epochs = 10;
    int nclass;
    int num_of_layers;
    std::vector<int> nhead;
    std::vector<int> num_features_per_layer;
    std::vector<GatLayer> attentions;
    bool verbose=true;

    friend class cereal::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version) {
    // Serialize member variables of GAT here (e.g., dropout, alpha, etc.)
    ar & dropout;
    ar & alpha;
    ar & num_epochs;
    ar & nclass;
    ar & num_of_layers;
    ar & nhead;
    ar & attentions;
    ar & num_features_per_layer;
    ar & verbose;
    }

    template <class Archive>
    void deserialize(Archive& ar, unsigned int version) {
    // Deserialize member variables of GAT here
    if (version > 1) {
        cout << "Première version" << endl;
    }
    ar(dropout);
    ar(alpha);
    ar(num_epochs);
    ar(nclass);
    ar(num_of_layers);
    ar(nhead);
    ar(attentions);
    ar(num_features_per_layer);
    ar(verbose);
    }
    

};

#endif