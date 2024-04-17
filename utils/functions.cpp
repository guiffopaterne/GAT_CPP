
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <unordered_map>
#include <vector>
#include "functions.hpp"

using namespace std;

Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits, int axis) {
    if (axis == 0) {
        // Softmax par colonnes (axis 0)
        Eigen::MatrixXd exp_logits = logits.array().exp();
        Eigen::RowVectorXd sum_exp_logits = exp_logits.colwise().sum();
        return exp_logits.array().rowwise() / sum_exp_logits.array();
    } else if (axis == 1) {
        // Softmax par lignes (axis 1)
        Eigen::MatrixXd exp_logits = logits.array().exp();
        Eigen::VectorXd sum_exp_logits = exp_logits.rowwise().sum();
        return exp_logits.array().colwise() / sum_exp_logits.array();
    } else {
        // Gestion d'une valeur d'axe incorrecte
        std::cerr << "Erreur : Axe incorrect. Utilisez 0 pour les colonnes, 1 pour les lignes." << std::endl;
        return Eigen::MatrixXd::Zero(logits.rows(), logits.cols());
    }
}

Eigen::MatrixXd encode_onehot(const Eigen::VectorXd& labels){
    Eigen::MatrixXd labels_encoder = Eigen::MatrixXd::Zero(labels.size(),2);
    for(int i = 0; i < labels.size(); ++i){
        if(labels(i)==1.0){
            labels_encoder(i,0) =1.0; 
        }else{
            labels_encoder(i,1) =1.0; 
        }   
    }
    return labels_encoder;
}

Eigen::MatrixXd convertMat2OneHot(const Eigen::MatrixXd& labels){
    return labels.unaryExpr([](double x) {return convert2OneHot(x);});

}

double convert2OneHot(double x) {
        return x > 0.5 ? 1.0 : 0.0;
    }


double accuracy(const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels, const Eigen::VectorXd& mask){
    cout<<"les dimensions output"<<output.cols()<<"x"<<output.rows()<<endl;
    cout<<"les dimensions labels"<<labels.cols()<<"x"<<labels.rows()<<endl;

    cout<<"Output"<<output.block(0,0,5,output.cols())<<endl;
    cout<<"Label "<<labels.block(0,0,5,labels.cols())<<endl;

    int correct_predictions=0,total_examples=0;
    for(int i = 0; i < labels.rows(); ++i){
        if(mask(i)==1.0){
            if (output.row(i) == labels.row(i)) {
                correct_predictions++;
            }

            total_examples++;
        }
    }
    return correct_predictions/total_examples;

}

Eigen::MatrixXd logSoftmax(const Eigen::MatrixXd& input, int axis) {
    return softmax(input,axis).array().log();
}

Eigen::MatrixXd nllLoss(const Eigen::MatrixXd& SoftmaxOutput, const Eigen::MatrixXd& target) {
    // Negative Log Likelihood Loss
    return -1*target*logSoftmax(SoftmaxOutput,1);
}

double loss(const Eigen::MatrixXd& output,const Eigen::VectorXd& mask){
    double l=0.0;
    Eigen::VectorXd row = output.rowwise().sum();
    for(int i = 0; i < output.rows(); ++i){
        if(mask(i)==1.0){
            l += row(i);
        }
    }
    return l;
}