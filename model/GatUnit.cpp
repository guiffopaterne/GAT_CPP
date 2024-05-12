
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <random>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <tuple>

#include "GatUnit.hpp"
#include "../utils/functions.hpp"
// #define Printer(VariableName) # VariableName
using namespace std;
using namespace Eigen;

    GatUnit::GatUnit(int in_features, int out_features ,double dropout, double alpha,string name,int nhead ,bool verbose)
        : in_features(in_features), out_features(out_features) ,dropout(dropout), alpha(alpha),name(name),nhead(nhead),verbose(verbose) {
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
        Eigen::MatrixXd h;
        if(isTrain)
            h = X.unaryExpr([this](double x) { return dropNode(x); });
        cout<<"les dimensions des H ="<<h.rows()<<"x"<<h.cols()<<" W   ="<< W.rows()<< "X"<<W.cols()<<endl;
        // calcule du node embedding
        Wh = h * W;
        if(isTrain)
            Wh = Wh.unaryExpr([this](double x) { return dropNode(x); });
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
        e.unaryExpr([this](double x) { return leakyReLU(x,alpha); });
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
        return h_prime;
    }

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> GatUnit::backward2(const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj,const Eigen::MatrixXd& d_h_prime) {
        // Gradients with respect to inputs (assuming leaky ReLU activation)
        shape(d_h_prime,"d_h_prime",verbose);
        shape(Wh,"Wh",verbose);
        if(d_h_prime.cols() != Wh.rows()){

        }
        Eigen::MatrixXd d_e  = d_h_prime * Wh.transpose();
        shape(d_e,"d_e",verbose);
        d_e = d_e.cwiseProduct(e.unaryExpr([this](double x){return leakyReLUPrime(x, alpha);}));  // Apply leaky ReLU derivative
        // Backpropagate through attention calculation
        Eigen::MatrixXd d_attention = d_h_prime * Wh.transpose();
        shape(d_attention,"d_attention",verbose);
        // Separate gradients for source and target nodes
        Eigen::MatrixXd d_Wh1 = d_attention.colwise().sum();
        Eigen::MatrixXd d_Wh2 = d_attention.rowwise().sum().transpose();
        shape(d_Wh1,"d_Wh1",verbose);
        shape(d_Wh2,"d_Wh2",verbose);
        // Backpropagation des scores d'attention
        Eigen::MatrixXd d_scoring_fn_source = d_Wh1*Wh;
        Eigen::MatrixXd d_scoring_fn_target = d_Wh2*Wh;
        shape(d_scoring_fn_source,"d_scoring_fn_source",verbose);
        shape(d_scoring_fn_target,"d_scoring_fn_target",verbose);
        // Braodcating d_Wh1 et d_Wh2 pour pouvoir les additionnes
        Eigen::MatrixXd d_Wh = d_Wh1.transpose().replicate(1, Wh.cols()) + d_Wh2.transpose().replicate(1, Wh.cols());
        // Multiplication element a element pour determiner le gradient de d_Wh(Node Embedding)
        d_Wh = d_Wh.cwiseProduct(scoring_fn_source.replicate(Wh.rows(), 1) + scoring_fn_target.replicate(Wh.rows(), 1));
        // shape(Wh1,"Wh1");
        // **Corrected d_W calculation (gradient w.r.t. weight matrix W):**
        shape(d_Wh,"d_Wh",verbose);
        shape(W,"W",verbose);
        Eigen::MatrixXd d_h = d_Wh * W.transpose();
        shape(d_h,"d_h",verbose);
        shape(X,"X",verbose);

        Eigen::MatrixXd d_W = X.transpose()*d_h_prime;

         // Backpropagate through node embedding
        //  Eigen::MatrixXd d_W = X.transpose() * d_h;
        shape(d_W,"d_w",verbose);
        // Gradient w.r.t. attention bias (assuming 'a' is the bias term)
        Eigen::MatrixXd d_a = d_h_prime.colwise().sum();
        shape(d_a,"d_a",verbose);
        // Return gradients as a tuple
        return std::make_tuple(d_h, d_W, d_a,d_scoring_fn_source, d_scoring_fn_target);
        }

   

   Eigen::MatrixXd GatUnit::update_parameters(const Eigen::MatrixXd& param , const Eigen::MatrixXd& grad,double lr,double beta1 ,double beta2 ,double epsilon) {
        // Mise à jour des paramètres en utilisant une optimiseur Adams 
        // Eigen::MatrixXd grad = gradient.colwise().mean(); // Moyenne sur le batch
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
        std::tie(grad_h,grad_W,grad_a, grad_source, grad_target) = backward2(h,adj,grad_output);
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

    // template <typename Archive>
    // void GatUnit::serialize(Archive& ar, const unsigned int version) {
    //     // Ajoutez ici la sérialisation/désérialisation des membres de votre classe GatUnit
    //     ar & W & a & dropout & alpha & concat & in_features & out_features; // Ajoutez les membres de GatUnit ici
    // }

    // template void GatUnit::serialize<boost::archive::text_iarchive>(boost::archive::text_iarchive&, const unsigned int);
    // template void GatUnit::serialize<boost::archive::text_oarchive>(boost::archive::text_oarchive&, const unsigned int);

        // std::tuple<Eigen::MatrixXd,Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> GatUnit::backward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output ) {
        // // Gradient w.r.t. attention (d_attention)
        // Eigen::MatrixXd d_attention = grad_output * Wh.transpose();
        // shape(d_attention,"d_attention",verbose);

        // // Gradient w.r.t. e (d_e)
        // Eigen::MatrixXd d_e = d_attention.array().cwiseProduct((attention.array() > 0).select(d_attention.array(), Eigen::MatrixXd::Zero(attention.rows(), attention.cols())));
        // d_e = d_e.unaryExpr([this](double x) { return leakyReLUPrime(x, alpha); });
        // shape(d_e,"d_e",verbose);
        // // Gradients w.r.t. Wh (d_Wh) and scoring functions (d_scoring_fn_target, d_scoring_fn_source)
        // Eigen::MatrixXd d_Wh = d_e.transpose() * attention;
        // shape(d_Wh,"d_Wh",verbose);
        // shape(Wh,"Wh",verbose);
        // Eigen::MatrixXd Wh1 = d_e * Eigen::RowVectorXd::Ones(d_e.cols()).transpose();
        // Eigen::MatrixXd Wh2 = d_e.transpose()* (Eigen::RowVectorXd::Ones(d_e.rows())).transpose();
        // shape(Wh1,"Wh1",verbose);
        // shape(Wh2,"Wh2",verbose);
        // // Gradient w.r.t. scoring_fn_source (d_scoring_fn_source)
        // Eigen::MatrixXd d_scoring_fn_source = Wh.transpose() * Wh1;
        // // Gradient w.r.t. scoring_fn_target (d_scoring_fn_target)
        // Eigen::MatrixXd d_scoring_fn_target = Wh.transpose() * Wh2;


        // // Eigen::MatrixXd d_scoring_fn_target = d_e.transpose() * Wh * Wh.transpose();
        // shape(d_scoring_fn_target,"d_scoring_fn_target",verbose);
        // // Eigen::MatrixXd d_scoring_fn_source = d_e.transpose() * Wh;
        // shape(d_scoring_fn_source,"d_scoring_fn_source",verbose);

        // // Gradient w.r.t. W (d_W)
        // Eigen::MatrixXd d_W = (h.transpose() * d_Wh.unaryExpr([this](double x) { return dropNode_prime(x,dropout); }));
        // shape(d_W,"d_W",verbose);
        // // Gradient w.r.t. bias (d_a)
        // Eigen::MatrixXd d_a;
        // if (a.rows()==h.rows()) {
        //     d_a = d_attention.colwise().sum();  // Sum across columns for element-wise bias
        // } else {
        //     d_a = grad_output.colwise().sum();  // Sum across columns for single bias term
        // }
        // // Eigen::MatrixXd d_a = grad_output.colwise().sum();
        // shape(d_a,"d_a",verbose);
        // shape(grad_output,"grad_output",verbose);
        // shape(W,"W",verbose);
        // // Return gradient w.r.t. input features (d_X)
        // Eigen::MatrixXd d_X = grad_output * W.unaryExpr([this](double x) { return dropNode_prime(x,dropout); }).transpose();
        // shape(d_X,"d_X",verbose);
        // return std::make_tuple(d_X,d_W,d_a,d_scoring_fn_source,d_scoring_fn_target);
        // }