#ifndef TRAIN_HPP
#define TRAIN_HPP

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <tuple>
#include <cmath>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>

#include "../model/GatNetwork.hpp"
#include "../utils/functions.hpp"

using namespace std;


class TRAIN {
public:
    TRAIN() = default; 
    TRAIN(int num_of_layers,vector<int> nhead,vector<int> num_features_per_layer,int nclass, double dropout, double alpha,bool verbose,int num_epochs,double lr,double beta1,double beta2,double epsilon,int patience, int early_stop,int num_thread, int num_thread_head);

    void excecute(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
                const Eigen::VectorXd& labels, const Eigen::VectorXi& train_mask, const Eigen::VectorXi& val_mask) ;
    void save_model(int epoch);
    GAT loadModel(string pathname);

    std::tuple<double,double> evaluate(const Eigen::MatrixXd& node_features, const Eigen::MatrixXd& adj,const Eigen::VectorXd& labels,const Eigen::VectorXi& test_mask) ;


private:
    double dropout;
    double alpha;
    int nclass;
    int num_of_layers;
    std::vector<int> nhead;
    std::vector<int> num_features_per_layer;
    bool verbose=true;
    int num_epochs;
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int patience;
    int early_stop;
    int num_thread;
    int num_thread_head =1;
    GAT gat;
};

#endif