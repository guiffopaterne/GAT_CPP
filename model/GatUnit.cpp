
#include <iostream>
#include <cmath>
#include <random>
#include <eigen3/Eigen/Dense>

 // Incluez avant Cereal Serialization
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>

#include "GatUnit.hpp"
#include "../utils/functions.hpp"
// #define Printer(VariableName) # VariableName
using namespace std;
using namespace Eigen;

    GatUnit::GatUnit(int in_features, int out_features ,double dropout, double alpha,bool concat,string name,int nhead ,bool verbose)
        : in_features(in_features), out_features(out_features) ,dropout(dropout), alpha(alpha),concat(concat),name(name),nhead(nhead),verbose(verbose) {
        initializeParameters();
    }

    Eigen::MatrixXd GatUnit::initialize_weights(int input_size, int output_size) {
        // Calcul de l'écart type selon l'initialisation Glorot
        float scale = std::sqrt(2.0 / (input_size + output_size));
        // Initialisation des poids selon une distribution normale centrée en zéro et
        // avec l'écart type calculé
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> distribution(0.0, scale);
        Eigen::MatrixXd weights(input_size, output_size);
        // Remplissage de la matrice avec les valeurs aléatoires
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weights(i, j) = distribution(gen);
            }
        }
        return weights;
    }

    void GatUnit::initializeParameters() {
        cout<<"\t Carateristique in_features="<<in_features<<", out_features"<<out_features<<endl;
        W = initialize_weights(in_features, out_features);
        a = Eigen::MatrixXd::Zero(1,out_features);
        scoring_fn_source = initialize_weights(1,out_features);
        scoring_fn_target = initialize_weights(1,out_features);
    }

    Eigen::MatrixXd GatUnit::forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj,bool isTrain=true) {
        if(isTrain){
            Wh = X.unaryExpr([this](double x) { return dropNode(x); })*W;
            Wh = Wh.unaryExpr([this](double x) { return dropNode(x); });
        }
        else Wh = X* W;
        shape(X,"X",verbose);
        shape(W,"W",verbose);
        // calcule du node embedding
        shape(Wh,"Wh",verbose);
        Eigen::MatrixXd Wh1 = Wh.cwiseProduct(scoring_fn_source.replicate(Wh.rows(),1));
        Eigen::MatrixXd Wh2 = Wh.cwiseProduct(scoring_fn_target.replicate(Wh.rows(),1));
        Wh1 =  Wh1.rowwise().sum();
        Wh2 = Wh2.rowwise().sum();
        Wh2= Wh2.transpose();
        shape(Wh1,"Wh1",verbose);
        shape(Wh2,"Wh2",verbose);
        // Eigen::MatrixXd e = Wh1.colwise() + Wh2.rowwise();
        e= Wh1.replicate(1,Wh.rows()) + Wh2.replicate(Wh.rows(),1);
        shape(e,"e",verbose);
        e = e.unaryExpr([this](double x) { return leakyReLU(x,alpha); });
        // preparation de la matrice d'attention en utilisant le scoring
        attention = e+adj;
        shape(attention,"attention",verbose);
        // normalisation des coefficients d'attention pour obtenir le poids des relations de chaque noeuds par rapport aux voisins
        cout<<" Calculer de l'attention encours"<<endl;
        attention = softmax(attention,1);
        cout<<" FIN du Calculer Calculer de l'attention "<<endl;
        Eigen::MatrixXd h_prime = attention*Wh;
        shape(h_prime,"h_prime",verbose);
        cout<<" h_prime agragation encours"<<endl;
        h_prime += a.replicate(Wh.rows(),1);
        if (concat) h_prime = h_prime.unaryExpr([this](double x){ return elu(x,alpha);});
        return h_prime;
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> GatUnit::backward(const Eigen::MatrixXd& X,const Eigen::MatrixXd& d_h_prime) {
        // Compute gradients for the skip connection
        Eigen::MatrixXd d_prime;
        if(concat) d_prime = d_h_prime.unaryExpr([this](double x) { return elu_prime(x, alpha); });
        else d_prime = d_h_prime;
        Eigen::MatrixXd d_a = d_prime.colwise().sum();
        // Compute gradients for Wh
        Eigen::MatrixXd d_Wh = attention.transpose() * d_prime;
        // Compute gradients for attention
        Eigen::MatrixXd d_attention = d_prime * Wh.transpose();
        // Compute gradients for e
        Eigen::MatrixXd d_e = d_attention.cwiseProduct(e.unaryExpr([this](double x) { return leakyReLUPrime(x, alpha); }));
        // Compute gradients for Wh1 and Wh2
        // Eigen::MatrixXd d_Wh1 = d_e.rowwise().sum().replicate(1, Wh.rows()).cwiseProduct(scoring_fn_source.replicate(Wh.rows(), 1));
        // Eigen::MatrixXd d_Wh2 = d_e.colwise().sum().replicate(Wh.rows(), 1).cwiseProduct(scoring_fn_target.replicate(Wh.rows(), 1));
        // // Compute gradients for scoring_fn_source and scoring_fn_target
        // Eigen::MatrixXd d_scoring_fn_source = (d_Wh1.cwiseProduct(Wh)).rowwise().sum();
        // Eigen::MatrixXd d_scoring_fn_target = (d_Wh2.cwiseProduct(Wh)).rowwise().sum();

        Eigen::MatrixXd d_Wh1 = d_e.rowwise().sum().replicate(1, Wh.cols()).cwiseProduct(scoring_fn_source.replicate(Wh.rows(), 1));
        Eigen::MatrixXd d_Wh2 = d_e.colwise().sum().transpose().replicate(1, Wh.rows()).cwiseProduct(scoring_fn_target.replicate(1, Wh.cols()));

        // Calcul du  gradients : scoring_fn_source et scoring_fn_target
        Eigen::MatrixXd d_scoring_fn_source = (d_Wh1.cwiseProduct(Wh)).colwise().sum();
        Eigen::MatrixXd d_scoring_fn_target = (d_Wh2.cwiseProduct(Wh)).rowwise().sum();

        // Compute gradients for W
        Eigen::MatrixXd d_X = d_Wh.unaryExpr([this](double x) { return dropNode_prime(x,dropout); });
        Eigen::MatrixXd d_W = (X.unaryExpr([this](double x) { return dropNode_prime(x,dropout); })).transpose() * d_X;
        // Compute gradients for X (d_h)
        Eigen::MatrixXd d_h = d_X * W.transpose();
        return std::make_tuple(d_h, d_W, d_a, d_scoring_fn_source, d_scoring_fn_target);
    }  

   Eigen::MatrixXd GatUnit::update_parameters(const Eigen::MatrixXd& param , const Eigen::MatrixXd& grad,double lr,double beta1 ,double beta2 ,double epsilon) {
        // Mise à jour des paramètres en utilisant une optimiseur Adams 
        Eigen::MatrixXd m = beta1 * param + (1 - beta1) * grad;
        Eigen::MatrixXd v = beta2 * param.array().square() + (1 - beta2) * grad.array().square();

        // Correction des biais
        double t = 1; // Numéro de l'itération
        Eigen::MatrixXd m_hat = m / (1 - std::pow(beta1, t+1));
        Eigen::MatrixXd v_hat = v / (1 - std::pow(beta2, t+1));

        // Mise à jour des paramètres
        Eigen::MatrixXd updated_temp= (lr * m_hat.array()) / (v_hat.array().sqrt() + epsilon);
        Eigen::MatrixXd updated_param = param - updated_temp;
        return updated_param;
    }
    Eigen::MatrixXd GatUnit::backward_update_parameters(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output,double lr,double beta1 ,double beta2 ,double epsilon) {
        // Mise à jour des paramètres
        Eigen::MatrixXd grad_h,grad_a, grad_source, grad_target,grad_W;
        // backward propagation
        std::tie(grad_h,grad_W,grad_a, grad_source, grad_target) = backward(h,grad_output);
        // mise a jour des poids W
        W = update_parameters(W, grad_W,lr,beta1,beta2,epsilon);
        // mise ajour des score d'attention
        scoring_fn_source = update_parameters(scoring_fn_source,grad_source,lr,beta1,beta2,epsilon);
        scoring_fn_target= update_parameters(scoring_fn_target,grad_target,lr,beta1,beta2,epsilon);
        // mise a jour du bias
        a  = update_parameters(a,grad_a,lr,beta1,beta2,epsilon);
        return grad_h;
    }
    double GatUnit::dropNode(double x) {
    // Génération d'un nombre aléatoire entre 0 et 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double random_value = distribution(gen);
    // Si la valeur aléatoire est inférieure à la probabilité de dropout, le neurone est "droppé"
    return  (random_value < dropout ) ? 0.00: x;
    }