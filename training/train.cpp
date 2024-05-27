
#include <iostream>
#include <fstream>
#include <filesystem>
#include <tuple>
#include <vector>
#include <cmath>

#include <sys/stat.h>
#include <sys/types.h>
#include <bits/stdc++.h>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Dense>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>

#include "train.hpp"
#include "../utils/functions.hpp"
#include "../model/GatNetwork.hpp"


using namespace std;
using namespace Eigen;
namespace fs = std::filesystem;

    TRAIN::TRAIN(int num_of_layers,vector<int> nhead,vector<int> num_features_per_layer,int nclass, double dropout, double alpha,bool verbose,int num_epochs,double lr,double beta1,double beta2,double epsilon,int patience, int early_stop):
                num_of_layers(num_of_layers) ,dropout(dropout),nhead(nhead),num_features_per_layer(num_features_per_layer),alpha(alpha), nclass(nclass),verbose(verbose),num_epochs(num_epochs), lr(lr),beta1(beta1),epsilon(epsilon),patience(patience),early_stop(early_stop){
        gat = GAT(num_of_layers,nhead,num_features_per_layer, nclass, dropout,alpha,verbose);

}

void TRAIN::save_model(int epoch){
    
    const string folder = "pretrained";
    const string filename = "gat_model"+ std::to_string(epoch)+".bin";
    
    fs::path p(folder);
    fs::path f(filename);
    if (!fs::exists(p)){
       fs::create_directory (p);
    }
    p =p / f;
    std::ofstream ofs(p.string());
    cereal::BinaryOutputArchive oa(ofs);
    oa(gat);
}
GAT TRAIN::loadModel(string pathname){
    GAT loadedModel;
    std::ifstream ifs(pathname);
    cereal::BinaryInputArchive ia(ifs);
  ia(loadedModel);
  return loadedModel;
}


void TRAIN::excecute(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,const Eigen::VectorXd& labels,const Eigen::VectorXi& train_mask, const Eigen::VectorXi& val_mask) {
    vector<double> accs;
    vector<double> loses;

    
    Eigen::MatrixXd features_sub,adj_sub ,features_val_sub,adj_val_sub;
    Eigen::VectorXd labels_sub,labels_val_sub;
    std::tie(features_sub,adj_sub,labels_sub)=create_sub_data(features,adjacency_matrix,labels,train_mask);
    std::tie(features_val_sub,adj_val_sub,labels_val_sub)=create_sub_data(features,adjacency_matrix,labels,val_mask);
    Eigen::MatrixXd label_sub_encoder = encode_onehot(labels_sub,nclass);

    

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Eigen::MatrixXd h =  gat.forward(features_sub,adj_sub,true);
        Eigen::MatrixXd grad_output = h - label_sub_encoder;        
        Eigen::MatrixXd out =  gat.backward(adj_sub,grad_output,lr,beta1,beta2,epsilon);
        std::cout << "fin backward "  << std::endl;
        double acc,loss;
        shape(features_val_sub,"features_val_sub",true);
        shape(adj_val_sub,"adj_val_sub",true);
        shape(labels_val_sub,"labels_val_sub",true);
        std::tie(acc,loss) = gat.evaluate(features_val_sub, adj_val_sub, labels_val_sub,val_mask);
        accs.push_back(acc);
        loses.push_back(loss);
        if(epoch%2==0 && epoch > 0){
            save_model(epoch);
        }
    }
}
std::tuple<double,double> TRAIN::evaluate(const Eigen::MatrixXd& node_features, const Eigen::MatrixXd& adj,const Eigen::VectorXd& labels,const Eigen::VectorXi& test_mask){
    Eigen::MatrixXd features_sub,adj_sub,labels_sub ;
    if(test_mask.size()> node_features.rows()){
        shape(node_features,"node_features",true);
        std::cerr<<"Impossible les tailles des matrix imcompatibles mask size = "<<test_mask.size()<<endl;
        return std::make_tuple(0.0,0.0);
    }
    if(test_mask.size()< node_features.rows())
            std::tie(features_sub,adj_sub,labels_sub)=create_sub_data(node_features,adj,labels,test_mask);
    else
        std::tie(features_sub,adj_sub,labels_sub)=std::make_tuple(node_features,adj,labels);
    return  gat.evaluate(features_sub, adj_sub, labels_sub,test_mask);
}
