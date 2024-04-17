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
    GAT(int nfeat, int nhid, int nclass, double dropout, double alpha, int nheads);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& adj);

     void backward(const Eigen::MatrixXd& x, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output);
     Eigen::VectorXd evaluate_node(const Eigen::MatrixXd& node_features, const Eigen::MatrixXd& adj);
    // void save_model(const std::string& filename);

    // // Fonction de chargement du modèle
    // void load_model(const std::string& filename);



    void train(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
                 const Eigen::MatrixXd& labels, const Eigen::VectorXd& train_mask, 
                 const Eigen::VectorXd& val_mask, int num_epochs);

   void train_batch(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
                 const Eigen::MatrixXd& labels, const Eigen::VectorXd& train_mask, 
                 const Eigen::VectorXd& val_mask, int batch_size, int num_epochs);
    void backward_batch(const Eigen::MatrixXd& x, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output);

    // Function to update parameters with averaged gradients for the batch
    void backward_batch_update_parameters(const Eigen::MatrixXd& grad_h_out, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& x);

    // Fonction d'évaluation sur l'ensemble de validation
    double evaluate(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
                    const Eigen::MatrixXd& labels, const Eigen::VectorXd& eval_mask);


    void set_num_epochs(int num_epochs);
    int nhid;


private:
    double dropout;
    double alpha;
    int nheads;
    int num_epochs = 10;
    int nclass;
    
    std::vector<GatLayer> attentions;
    GatLayer out_att;

    double dropNode(double x);
    double elu(double x);

    // friend class boost::serialization::access;

    // template <typename Archive>
    // void serialize(Archive& ar, const unsigned int version) ;
};

#endif