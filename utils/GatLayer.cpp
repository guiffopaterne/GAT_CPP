#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <tuple>

#include "GatLayer.hpp"
#include "functions.hpp"
using namespace std;
using namespace Eigen;

    GatLayer::GatLayer(int in_features, int out_features, double dropout, double alpha, bool concat)
        : in_features(in_features), out_features(out_features), dropout(dropout), alpha(alpha), concat(concat) {
        W.resize(in_features, out_features);
        a.resize(2 * out_features, 1);
        initializeParameters();
    }

    Eigen::MatrixXd GatLayer::forward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj) {
        cout<<"les dimensions des H ="<<h.rows()<<"x"<<h.cols()<<" W  rows ="<< W.rows()<< " W cols ="<<W.cols()<<endl;
        Eigen::MatrixXd Wh = h * W; // h.shape: (N, in_features), Wh.shape: (N, out_features)
        Eigen::MatrixXd e = prepareAttentionalMechanismInput(Wh);
        // cout<<"les dimensions des HW ="<<Wh.rows()<<"X "<<Wh.cols()<<" e ="<< e.rows()<< "X"<<e.cols()<<endl;
        Eigen::MatrixXd attention = -9e15 * Eigen::MatrixXd::Ones(e.rows(),e.cols());
        attention = (adj.array()>0).select(e,attention);
        attention = softmax(attention,1);
        attention = attention.unaryExpr([this](double x) { return dropNode(x); });

        Eigen::MatrixXd hPrime = attention*Wh;
        cout<<"les dimensions des hPrime ="<<hPrime.rows()<<"x"<<hPrime.cols()<<endl;
        if (concat) {
            return hPrime.unaryExpr([this](double x) { return elu(x); });
        } else {
            return hPrime;
        }
    }
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> GatLayer::backward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& gradput) {
        Eigen::MatrixXd Wh = h*W;
        Eigen::MatrixXd e = prepareAttentionalMechanismInput(Wh);
        Eigen::MatrixXd attention = -9e15 * Eigen::MatrixXd::Ones(e.rows(),e.cols());
        attention = (adj.array()>0).select(e,attention);
        attention = softmax(attention,1);
        attention = attention.unaryExpr([this](double x) { return dropNode(x); });

        Eigen::MatrixXd hPrime = attention * Wh;

        // Gradients with respect to Wh
        Eigen::MatrixXd grad_Wh = gradput * attention;

        // Gradients with respect to attention
        cout<<"les dimensions de gradput="<<gradput.rows()<<"x"<<gradput.cols()<<" W="<< Wh.rows()<<"x"<<Wh.cols()<<endl;

        Eigen::MatrixXd grad_attention = gradput*Wh;

        // Gradients with respect to a
        Eigen::MatrixXd grad_a = grad_attention* prepareAttentionalMechanismInputPrime(Wh);
        grad_a = grad_a.transpose() * h;

        // Gradients with respect to W
        Eigen::MatrixXd grad_W = grad_Wh.transpose() * h;
        cout<<"les dimensions des grad_Wh ="<<grad_Wh.rows()<<"x"<<grad_Wh.cols()<<" W  rows ="<< W.rows()<< " W cols ="<<W.cols()<<endl;
        // Gradients with respect to h
        Eigen::MatrixXd grad_h = gradput * (W.transpose() * grad_Wh);
        backward_update_parameters(h, grad_h);

        return std::make_tuple(grad_W, grad_a, grad_h);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> GatLayer::backward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& gradput) {
        // Calcul des gradients pour la couche de sortie
        Eigen::MatrixXd grad_h = gradput;
        Eigen::MatrixXd grad_Wh = attention_backward(grad_h);

        // Mise à jour des paramètres de la couche de sortie
        backward_update_parameters(h, grad_h);

        return std::make_tuple(grad_h, grad_Wh, W);
    }

   Eigen::MatrixXd GatLayer::update_parameters(const Eigen::MatrixXd& param, const Eigen::MatrixXd& input, const Eigen::MatrixXd& gradient) {
        // Mise à jour des paramètres en utilisant une règle de mise à jour spécifique (par exemple, Adam)
        // Adapté en fonction de vos besoins spécifiques
        // Pseudo-code, vous devrez adapter cette fonction selon votre algorithme d'optimisation
        double learning_rate = 0.01; // Paramètre d'apprentissage
        double beta1 = 0.9; // Moment d'ordre 1
        double beta2 = 0.999; // Moment d'ordre 2
        double epsilon = 1e-8; // Petit terme pour éviter la division par zéro

        Eigen::MatrixXd grad = gradient.colwise().mean(); // Moyenne sur le batch
        Eigen::MatrixXd m = beta1 * param + (1 - beta1) * grad;
        Eigen::MatrixXd v = beta2 * param.array().square() + (1 - beta2) * grad.array().square();

        // Correction des biais
        double t = 1; // Numéro de l'itération
        Eigen::MatrixXd m_hat = m / (1 - std::pow(beta1, t));
        Eigen::MatrixXd v_hat = v / (1 - std::pow(beta2, t));

        // Mise à jour des paramètres
        Eigen::MatrixXd updated_temp= (learning_rate * m_hat.array()) / (v_hat.array().sqrt() + epsilon);
        Eigen::MatrixXd updated_param = param - updated_temp;

        return updated_param;
    }
    void GatLayer::backward_update_parameters(const Eigen::MatrixXd& h, const Eigen::MatrixXd& grad_h) {
        // Mise à jour des paramètres pour les poids W de la couche de sortie
        W = update_parameters(W, h, grad_h);
        // Mise à jour des paramètres pour les poids a de la couche de sortie
        a = update_parameters(a, h, grad_h);
    }
    double GatLayer::elu(double x) {
        return x > 0 ? x : alpha * (std::exp(x) - 1);
    }



    void GatLayer::initializeParameters() {
        W = Eigen::MatrixXd::Random(in_features, out_features);
        a = Eigen::MatrixXd::Random(2 * out_features, 1);
    }

    Eigen::MatrixXd GatLayer::prepareAttentionalMechanismInput(const Eigen::MatrixXd& Wh) {
        Eigen::MatrixXd Wh1 = Wh * a.topRows(out_features);
        Eigen::MatrixXd Wh2 = Wh * a.bottomRows(out_features);
        // cout<< "la taille de Wh1="<<Wh1.rows()<<"x"<<Wh1.cols()<< " Wh2 "<<Wh2.rows()<<"x"<<Wh2.cols()<<endl;
        Wh2 = Wh2 * Eigen::RowVectorXd::Ones(Wh.rows());
        Wh1= Wh1 * Eigen::RowVectorXd::Ones(Wh.rows());
        // cout<< "la taille de Wh1="<<Wh1.rows()<<"x"<<Wh1.cols()<< " Wh2 "<<Wh2.rows()<<"x"<<Wh2.cols()<<endl;
        Eigen::MatrixXd e = Wh1.array() + Wh2.transpose().array();
        // cout<< "la taille de Wh1="<<Wh1.rows()<<"x"<<Wh1.cols()<< " Wh2 "<<Wh2.rows()<<"x"<<Wh2.cols()<<" "<<endl;
        e.unaryExpr([this](double x) { return leakyReLU(x); });
        return e;
    }

    double GatLayer::leakyReLU(double x) {
        return x > 0 ? x : alpha * x;
    }

    double GatLayer::dropNode(double x) {
        return (std::rand() / static_cast<double>(RAND_MAX)) > dropout ? x / (1 - dropout) : 0.0;
    }

    
     Eigen::MatrixXd GatLayer::prepareAttentionalMechanismInputPrime(const Eigen::MatrixXd& Wh) {
        return Wh.unaryExpr([this](double x) { return leakyReLUPrime(x); });
    }

    double GatLayer::leakyReLUPrime(double x) {
        return x > 0 ? 1.0 : alpha;
    }
    
    Eigen::MatrixXd GatLayer::attention_backward(const Eigen::MatrixXd& grad_h) {
        // Calcul des gradients pour le mécanisme attentionnel lors de la rétropropagation
        Eigen::MatrixXd grad_Wh = grad_h * attention_derivative();
        return grad_Wh;
    }

    Eigen::MatrixXd GatLayer::attention_derivative() {
        // Calcul de la dérivée de la fonction d'attention
        return a.topRows(out_features) * a.bottomRows(out_features).transpose();
    }

    // template <typename Archive>
    // void GatLayer::serialize(Archive& ar, const unsigned int version) {
    //     // Ajoutez ici la sérialisation/désérialisation des membres de votre classe GatLayer
    //     ar & W & a & dropout & alpha & concat & in_features & out_features; // Ajoutez les membres de GatLayer ici
    // }

    // template void GatLayer::serialize<boost::archive::text_iarchive>(boost::archive::text_iarchive&, const unsigned int);
    // template void GatLayer::serialize<boost::archive::text_oarchive>(boost::archive::text_oarchive&, const unsigned int);