#include <iostream>
#include <eigen3/Eigen/Dense>
 // Incluez avant Boost Serialization
#include <boost/serialization/vector.hpp>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <tuple>
#include <cmath>
#include <bits/stdc++.h>
#include "GatLayer.hpp"
#include "GatNetwork.hpp"
#include "../utils/vector_serialization.hpp" 
#include "../utils/functions.hpp"

using namespace std;


GAT::GAT(int num_of_layers, std::vector<int> nhead,std::vector<int> num_features_per_layer, int nclass, double dropout, double alpha,bool verbose)
    : dropout(dropout),nhead(nhead),num_features_per_layer(num_features_per_layer),alpha(alpha), nclass(nclass),verbose(verbose) {
        
    

    // attentions.push_back(GatLayer(nfeat, 8*nhead[0], dropout, alpha, true));
    if(num_of_layers==nhead.size() and num_of_layers==num_features_per_layer.size()){
        cout<<"entrer les information valide pour la formation \n"<<endl;
    }
    // nhead.insert(nhead.begin()+1,1);
    for (int i = 0; i < num_of_layers; ++i) {
        cout<<"Layer "<<i+1<<" variable in_features="<<num_features_per_layer[i]*nhead[i]<<"out_features= "<< num_features_per_layer[i+1]<< " num_of_heads="<<nhead[i+1] <<endl;
        // GatLayer(int in_features, int out_features,int num_of_heads ,double dropout, double alpha, bool concat)
        attentions.push_back(GatLayer(
            num_features_per_layer[i]*nhead[i],
            num_features_per_layer[i+1],
            nhead[i+1],dropout, alpha, 
            (i < num_of_layers - 1)?true:false,"Layer "+to_string(i+1),
            verbose));
    }
}

Eigen::MatrixXd GAT::evaluate_forward(const Eigen::MatrixXd& X,const Eigen::MatrixXd& adj){
    Eigen::MatrixXd r = attentions[0].forward(X,adj,false);
    for (int i = 1; i < attentions.size(); ++i)
        r = attentions[i].forward(r,adj,false);
    return r;
}

Eigen::MatrixXd GAT::forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj){
    // X = X.unaryExpr([this](double x) { return dropNode(x,dropout); });
    int el = 1;
    int numAttentions = attentions.size();
    cout<<" forward debut layer "<<el;
    Eigen::MatrixXd r = attentions[el-1].forward(X, adj,true);
    el++;
    cout<<"FIN 1"<<endl;
    shape(r,"r",verbose);
    for (int i = 1; i < numAttentions; ++i) {
        cout<<" forward debut layer.."<<el;
        r = attentions[i].forward(r, adj,true);
        cout<<" FIN layer "<<el<<endl;
        shape(r,"r",verbose);
        el++;
    }
    cout<<" activation.... "<<el-1;
    cout<<"FIN "<<endl;
    return r;
    }

Eigen::MatrixXd GAT::backward(const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output,double lr,double beta1 ,double beta2 ,double epsilon) {
    // int numLayer = -1;
    Eigen::MatrixXd grad_h = grad_output;
    //  attentions[numLayer].backward(adj, grad_output,lr,beta1,beta2,epsilon);
    int i = attentions.size();
    // Backward pass through each attention layer
    int el=1;
    while(i>0){
        cout<<"couche n-"<<i<<endl;
        grad_h = attentions[i-1].backward(adj, grad_h,lr,beta1,beta2,epsilon);
        i--;
    }
    return grad_h;
}

std::tuple<double,double> GAT::evaluate(const Eigen::MatrixXd& node_features, const Eigen::MatrixXd& adj,const Eigen::VectorXd& labels ,const Eigen::VectorXi& val_mask) {
    // Forward pass through the GAT for a single node
    // h = h.unaryExpr([this](double x) { return dropNode(x); });
    Eigen::MatrixXd label_encoder= encode_onehot(labels,nclass);
    Eigen::MatrixXd h = evaluate_forward(node_features,adj);
    double acc = accuracy(argmax(h),labels,val_mask);
    cout <<"Accurancy "<<acc<<endl;
    double l = nllLoss(h,label_encoder,val_mask);
    std::cout <<"Loss: " << l << std::endl;
    return std::make_tuple(acc,l);
}

// string describe(){
//     string layer_describe = " ";
//     for()
// }


void GAT::train(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
                const Eigen::VectorXd& labels, const Eigen::VectorXi& train_mask, const Eigen::VectorXi& val_mask,
                int num_epochs,double lr,double beta1,double beta2,double epsilon,int patience, int early_stop) {
    vector<double> accs;
    vector<double> loses;

    
    Eigen::MatrixXd features_sub,adj_sub ,features_val_sub,adj_val_sub;
    Eigen::VectorXd labels_sub,labels_val_sub;
    std::tie(features_sub,adj_sub,labels_sub)=create_sub_data(features,adjacency_matrix,labels,train_mask);
    std::tie(features_val_sub,adj_val_sub,labels_val_sub)=create_sub_data(features,adjacency_matrix,labels,val_mask);
    Eigen::MatrixXd label_sub_encoder = encode_onehot(labels_sub,nclass);


    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Eigen::MatrixXd h =  forward(features_sub,adj_sub);
        Eigen::MatrixXd grad_output = h - label_sub_encoder;        
        Eigen::MatrixXd out =  backward(adj_sub,grad_output,lr,beta1,beta2,epsilon);
        std::cout << "fin backward "  << std::endl;
        double acc,loss;

        std::tie(acc,loss) = evaluate(features_val_sub, adj_val_sub, labels, val_mask);
        accs.push_back(acc);
        loses.push_back(loss);
    }
}


void GAT::set_num_epochs(int num_epochs){
    num_epochs=num_epochs;
}

string GAT::print_representation(){
        string str= "GAT(Number of Layers: " + to_string(num_of_layers)+ ",Hidden Units: "+ to_string(num_features_per_layer.size()-2)
        +",\n Number of Classes: " + to_string(nclass) +"- Dropout Rate: " + to_string(dropout)+"- Alpha (LeakyReLU Slope): " + to_string(alpha);
        return str;
    }

// void GAT::save_model(const std::string& filename){
//     std::ofstream ofs(filename);
//     boost::archive::text_oarchive ar(ofs);
//     ar << *this; // Utilisez la sérialisation pour sauvegarder la classe GAT
// }

// // Fonction de chargement du modèle
// void GAT::load_model(const std::string& filename) {
//     std::ifstream ifs(filename);
//     boost::archive::text_iarchive ar(ifs);
//     ar >> *this; // Utilisez la désérialisation pour charger la classe GAT
// }


// template <typename Archive>
// void GAT::serialize(Archive& ar, const unsigned int version) {
//     // Ajoutez ici la sérialisation/désérialisation des membres de votre classe GAT
//    ar & num_epochs & dropout & alpha & nheads & attentions & out_att;
// }
