#include <iostream>
#include <fstream>
#include <cmath>
#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>

 // Incluez avant Cereal Serialization
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>

#include "GatLayer.hpp"
#include "GatNetwork.hpp"
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


Eigen::MatrixXd GAT::forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj,int num_thread,int num_thread_head,bool isTrain=true){
    int el = 1;
    int numAttentions = attentions.size();
    if(isTrain)cout<<" forward debut layer "<<el;
    else cout<<" EVALUATION  layer "<<el;
    Eigen::MatrixXd r = X;
    shape(r,"r",verbose);
    for (int i = 0; i < numAttentions; ++i) {
        cout<<" forward debut layer.."<<el;
        r = attentions[i].forward(r, adj, num_thread_head,isTrain);
        cout<<" FIN layer "<<el<<endl;
        shape(r,"r",verbose);
    }
    cout<<"FIN "<< el-1<<endl;
    return r;
    }

Eigen::MatrixXd GAT::backward(const Eigen::MatrixXd& adj, int num_thread,int num_thread_head,const Eigen::MatrixXd& grad_output,double lr,double beta1 ,double beta2 ,double epsilon) {
    Eigen::MatrixXd grad_h = grad_output;
    int i = attentions.size();
    // Backward pass through each attention layer
    int el=1;
    while(i>0){
        cout<<"couche n-"<<i<<endl;
        grad_h = attentions[i-1].backward(adj, num_thread_head,grad_h,lr,beta1,beta2,epsilon);
        i--;
    }
    return grad_h;
}

std::tuple<double,double> GAT::evaluate(const Eigen::MatrixXd& node_features, const Eigen::MatrixXd& adj,int num_thread,const Eigen::VectorXd& labels ,const Eigen::VectorXi& val_mask) {
    double acc=0.0,l=0.0;
    Eigen::MatrixXd label_encoder= encode_onehot(labels,nclass);
    Eigen::MatrixXd h = forward(node_features,adj,num_thread,false);

    if(labels.size()==val_mask.size()){
        acc = accuracy(argmax(h),labels);
        l = nllLoss(h,label_encoder);
    }else{
        acc = accuracy(argmax(h),labels,val_mask);
        l = nllLoss(h,label_encoder,val_mask);
    }
    cout <<"Accurancy "<<acc<<endl;
    std::cout <<"Loss: " << l << std::endl;
    return std::make_tuple(acc,l);
}



// void GAT::train(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
//                 const Eigen::VectorXd& labels, const Eigen::VectorXi& train_mask, const Eigen::VectorXi& val_mask,
//                 int num_epochs,double lr,double beta1,double beta2,double epsilon,int patience, int early_stop) {
//     std::vector<double> accs;
//     std::vector<double> loses;

    
//     Eigen::MatrixXd features_sub,adj_sub ,features_val_sub,adj_val_sub;
//     Eigen::VectorXd labels_sub,labels_val_sub;
//     std::tie(features_sub,adj_sub,labels_sub)=create_sub_data(features,adjacency_matrix,labels,train_mask);
//     std::tie(features_val_sub,adj_val_sub,labels_val_sub)=create_sub_data(features,adjacency_matrix,labels,val_mask);
//     Eigen::MatrixXd label_sub_encoder = encode_onehot(labels_sub,nclass);


//     for (int epoch = 0; epoch < num_epochs; ++epoch) {
//         Eigen::MatrixXd h =  forward(features_sub,adj_sub,1,1);
//         Eigen::MatrixXd grad_output = h - label_sub_encoder;        
//         Eigen::MatrixXd out =  backward(adj_sub,1,1,grad_output,lr,beta1,beta2,epsilon);
//         std::cout << "fin backward "  << std::endl;
//         double acc,loss;

//         std::tie(acc,loss) = evaluate(features_val_sub, adj_val_sub,1 , labels, val_mask);
//         accs.push_back(acc);
//         loses.push_back(loss);
//     }
// }


void GAT::set_num_epochs(int num_epochs){
    num_epochs=num_epochs;
}

string GAT::print_representation(){
        string str= "GAT(Number of Layers: " + to_string(num_of_layers)+ ",Hidden Units: "+ to_string(num_features_per_layer.size()-2)
        +",\n Number of Classes: " + to_string(nclass) +"- Dropout Rate: " + to_string(dropout)+"- Alpha (LeakyReLU Slope): " + to_string(alpha);
        return str;
    }

// ensemble de getter
    int GAT::get_num_epoch(){
        return num_epochs;
    }

    // int get_num_epochs()

    int GAT::get_nclass(){
        return nclass;
    }
    int GAT::get_num_of_layers(){
        return attentions.size();
    }
    std::vector<int> GAT::get_num_head_per_layer(){
        return nhead;
    }
    std::vector<int> GAT::get_num_features_per_layer(){
        return num_features_per_layer;
    }
    std::vector<GatLayer> GAT::get_attentions(){
        return attentions;
    }