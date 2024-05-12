#ifndef GAT_NETWORK_HPP
#define GAT_NETWORK_HPP
#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <tuple>
#include <cmath>
#include "GatLayer.hpp"

using namespace std;


class GAT {
public:
    GAT() = default; 
    GAT(int num_of_layers,vector<int> nhead,vector<int> num_features_per_layer,int nclass, double dropout, double alpha,bool verbose);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& adj);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output,double lr,double beta1 ,double beta2 ,double epsilon);
    Eigen::MatrixXd evaluate_forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj);
    string print_representation();



    void train(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
                const Eigen::VectorXd& labels, const Eigen::VectorXi& train_mask, const Eigen::VectorXi& val_mask,
                int num_epochs,double lr,double beta1,double beta2,double epsilon,int patience, int early_stop) ;

    std::tuple<double,double> evaluate(const Eigen::MatrixXd& node_features, const Eigen::MatrixXd& adj,const Eigen::VectorXd& labels,const Eigen::VectorXi& val_mask) ;


    void set_num_epochs(int num_epochs);


private:
    double dropout;
    double alpha;
    int num_epochs = 10;
    int nclass;
    int num_of_layers;
    std::vector<int> nhead;
    std::vector<int> num_features_per_layer;
    std::vector<GatLayer> attentions;
    bool verbose=true;

    // friend class boost::serialization::access;

    // template <typename Archive>
    // void serialize(Archive& ar, const unsigned int version) ;

    // void save_model(const std::string& filename);

    // // Fonction de chargement du modèle
    // void load_model(const std::string& filename);
};

#endif