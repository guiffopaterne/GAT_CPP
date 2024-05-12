
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <unordered_map>
#include <vector>
#include <tuple>
#include "functions.hpp"

using namespace std;
using namespace Eigen;

Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits,const int axis) {
    
    if((axis != 1 && axis != 0)){
        std::invalid_argument("Erreur : Axe incorrect. Utilisez 0 pour les colonnes, 1 pour les lignes." );
    }
    Eigen::MatrixXd exp_logits = logits.array().exp();
    if (axis == 1)
        exp_logits.array().rowwise().normalize();
    else 
        exp_logits.array().colwise().normalize();
    return exp_logits;
    
}

Eigen::MatrixXd encode_onehot(const Eigen::VectorXd& labels,int num_class){
    Eigen::MatrixXd labels_encoder = Eigen::MatrixXd::Zero(labels.size(),num_class);
    for(int i = 0; i < labels.size(); ++i){
        labels_encoder(i,(int)labels(i))=1.0;  
    }
    return labels_encoder;
}

Eigen::MatrixXd encode_onehot_out(const Eigen::MatrixXd& output){
    Eigen::MatrixXd labels_encoder = Eigen::MatrixXd::Zero(output.rows(),output.cols());
    for(int i =0;i< output.rows();i++){
        labels_encoder(i,argmax_vector(output.row(i)))=1;
    }
    return labels_encoder;
}

Eigen::VectorXd argmax(Eigen::MatrixXd m){
    Eigen::VectorXd v = Eigen::VectorXd::Zero(m.rows());
    for(int i=0;i<m.rows();i++){
        v(i)=argmax_vector(m.row(i));
    }
    return v;
}

int argmax_vector(const Eigen::VectorXd& v){
    int max_index =0;
    double max = v(0);
    for(int j= 0;j<v.size();j++){
        if(max>v(j)){
            max = v(j);
            max_index=j;
        }
    }
    return max_index;
}

Eigen::MatrixXd convertMat2OneHot(const Eigen::MatrixXd& labels){
    return labels.unaryExpr([](double x) {return convert2OneHot(x);});
}

double convert2OneHot(const double x) {
        return x > 0.5 ? 1.0 : 0.;
}
void head(const Eigen::MatrixXd& M,const string name,int ligne=5){
    cout<<name<<M.block(0,0,ligne,M.cols())<<endl;
}


double accuracy(const Eigen::VectorXd& output, const Eigen::VectorXd& labels, const Eigen::VectorXi& mask){
    head(output,"output");
    head(labels,"labels");
    int correct_predictions=0;
    int total_examples=mask.size();
    for(auto i :mask)
        if(output(i)==labels(i)) 
            correct_predictions++;
    return correct_predictions/total_examples;
}

Eigen::MatrixXd logSoftmax(const Eigen::MatrixXd& input, int axis) {
    return softmax(input,axis).array().log();
}

Eigen::MatrixXd logSoftmax(const Eigen::MatrixXd& input) {
    return input.array().log();
}

double nllLoss(const Eigen::MatrixXd& SoftmaxOutput, const Eigen::MatrixXd& target,const Eigen::VectorXi& mask) {
    if(SoftmaxOutput.rows()!= target.rows() || SoftmaxOutput.cols()!= target.cols()){
        std::invalid_argument("Erreur : Axe incorrect. Utilisez 0 pour les colonnes, 1 pour les lignes." );
    }
    // Negative Log Likelihood Loss
    Eigen::MatrixXd x = -1*target.cwiseProduct(logSoftmax(SoftmaxOutput));
    int sum=0;
    for(auto & i : mask) sum+=x.row(i).sum();
    return sum/mask.size();
}
double nllLoss(const Eigen::MatrixXd& Output, const Eigen::MatrixXd& target,const Eigen::VectorXi& mask,int axis) {
    // Negative Log Likelihood Loss
    Eigen::MatrixXd x = -1*target.cwiseProduct(logSoftmax(Output,axis));
    int sum=0;
    for(auto & i : mask) sum+=x.row(i).sum();
    return sum/mask.size();
}
double loss(const Eigen::MatrixXd& output,const Eigen::VectorXi& mask){
    double l=0.0;
    Eigen::VectorXd row = output.rowwise().sum();
    for(int i = 0; i < mask.size(); ++i){
            l += row(mask(i));
    }
    return l;
}

std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::VectorXi> createMask(const int size,const int train_size, const int val_size, bool isShuffle=false) {
    Eigen::VectorXi train_mask= Eigen::VectorXi::Zero(train_size);
    Eigen::VectorXi val_mask = Eigen::VectorXi::Zero(val_size);
    Eigen::VectorXi test_mask = Eigen::VectorXi::Zero(size-val_size-train_size);
    Eigen::VectorXi mask = Eigen::VectorXi::Zero(size);
    for(int i=0;i<size;++i) mask[i]=i;
    if(isShuffle){
    std::random_shuffle(mask.begin(),mask.end());
    }
    for(int i=0;i<train_size;++i) train_mask(i)=mask(i);
    for(int i=train_size,j=0;i<train_size+val_size;++i,++j) val_mask(j)=mask(i);
    for(int i=train_size+val_size,j=0;i<size;++i,++j) train_mask(j)=mask(i);
    return std::make_tuple(train_mask,val_mask,test_mask);
}

Eigen::MatrixXd create_sub_adjacency_matrix(const Eigen::MatrixXd& adj_matrix, const Eigen::VectorXi& mask) {
  // Validate input
  if (mask.size() > adj_matrix.rows()) {
    throw std::invalid_argument("la taille du mask ne peut pas depasse la matrix d'origine.");
  }
  // recuperer la taille du mask
  int sub_matrix_size = mask.size();
  // construction de la sous matrice adjence
  Eigen::MatrixXd sub_adj_matrix(sub_matrix_size, sub_matrix_size);
  // Extraire les ligne et les colonne qui corresponde dans la matrice initale
  for (int i = 0; i < sub_matrix_size; ++i) {
    int row_index = mask[i];
    for (int j = 0; j < sub_matrix_size; ++j) {
      int col_index = mask[j];
      sub_adj_matrix(i, j) = adj_matrix(row_index, col_index);
    }
  }
  return sub_adj_matrix;
}
Eigen::MatrixXd create_sub_features_matrix(const Eigen::MatrixXd& features_matrix, const Eigen::VectorXi& mask) {
  // Validate input
  if (mask.size() > features_matrix.rows()) {
    throw std::invalid_argument("la taille du mask ne peut pas depasse la matrix d'origine.");
  }
  int num_rows = mask.size();
  Eigen::MatrixXd sub_features_matrix(num_rows, features_matrix.cols());
  for (int i = 0; i < num_rows; ++i) {
    int row_index = mask(i);
    if (row_index >= 0 && row_index < features_matrix.rows()) {
      sub_features_matrix.row(i) = features_matrix.row(row_index);
    } else {
      std::cerr<<"Ligne "<< row_index << "Introvable"<<endl;
    }
  }

  return sub_features_matrix;
}

Eigen::VectorXd create_sub_labels_vector(const Eigen::VectorXd& labels_vector, const Eigen::VectorXi& mask) {
  // Validate input
  if (mask.size() > labels_vector.size()) {
    throw std::invalid_argument("la taille du mask ne peut pas depasse la matrix d'origine.");
  }
  int num_rows = mask.size();
  Eigen::VectorXd sub_labels_vector(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    int row_index = mask(i);
    if (row_index >= 0 && row_index < labels_vector.size()) {
      sub_labels_vector(i) = labels_vector(row_index);
    } else {
      std::cerr<<"Ligne "<< row_index << "Introvable"<<endl;
    }
  }

  return sub_labels_vector;
}

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::VectorXd> create_sub_data(const Eigen::MatrixXd& features_matrix ,const Eigen::MatrixXd& adj_matrix,const Eigen::VectorXd& labels_vector, const Eigen::VectorXi& mask){
    Eigen::MatrixXd sub_features = create_sub_features_matrix(features_matrix,mask);
    Eigen::MatrixXd adj_sub = create_sub_adjacency_matrix(adj_matrix,mask);
    Eigen::VectorXd labels_sub = create_sub_labels_vector(labels_vector,mask);  
    return  std::make_tuple(sub_features,adj_sub,labels_sub);

}

double elu(const double x,const double alpha) {
        return x > 0 ? x : alpha * (std::exp(x) - 1);
    }

double leakyReLU(const double x,const double alpha) {
        return x > 0 ? x : alpha * x;
    }
double leakyReLUPrime(const double x,const double alpha) {
        return x > 0 ? 1.0 : alpha;
    }
double dropNode_prime(const double x, const double dropout) {
    return (x != 0)? 1.0 / (1.0 - dropout): 0.0;
    }
double dropNode(double x,const double dropout) {
        return (std::rand() / static_cast<double>(RAND_MAX)) > dropout ? x / (1 - dropout) : 0.0;
    }
Eigen::MatrixXd softmax_prime(const Eigen::MatrixXd& S) {
    return S.array() * (-1*S.array()+1);
}

void shape(const Eigen::MatrixXd& m,const string name,bool verbose=true){
    if(verbose) cout<<"Dimension de "<<name<<"="<<m.rows()<<"x"<<m.cols()<<endl;
}