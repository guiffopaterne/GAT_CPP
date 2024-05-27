#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <random>

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>

#include "GatLayer.hpp"
#include "functions.hpp"
#include "GatUnit.hpp"

using namespace std; 
using namespace Eigen;

    GatLayer::GatLayer(int in_features, int out_features,int num_of_heads ,double dropout, double alpha, bool concat,string name,bool verbose)
        : in_features(in_features), out_features(out_features),num_of_heads(num_of_heads) ,dropout(dropout), alpha(alpha), concat(concat), name(name),verbose(verbose){
         for (int i = 0; i < num_of_heads; ++i) {
             cout<<"Head ="<<i+1<<" out_features"<<endl;
            GatUnit head_unit(in_features, out_features, dropout, alpha,name+"HEAD -"+to_string(i+1),i,verbose);
            layer.push_back(head_unit);
            }
    }

    Eigen::MatrixXd GatLayer::stack_heads(const std::vector<Eigen::MatrixXd>& head_outputs) {
        // Initialiser la matrice résultante en fonction du mode de stacking (concaténation ou addition)
        int num_nodes = head_outputs[0].rows(); // Nombre de nœuds
        int num_features_per_head = head_outputs[0].cols(); // Nombre de caractéristiques par tête
        int total_num_features;

        Eigen::MatrixXd stacked_output; // Matrice résultante qui empile les sorties de chaque tête
        total_num_features = concat?num_of_heads * num_features_per_head:num_features_per_head;
        stacked_output.resize(num_nodes, total_num_features); // Initialiser la matrice résultante pour l'addition
        stacked_output.setZero();
        // Empiler les sorties de chaque tête dans la matrice résultante
        for (int i = 0; i < num_of_heads; ++i) {
            if (concat) {
                stacked_output.block(0, i * num_features_per_head, num_nodes, num_features_per_head) = head_outputs[i];
            } else {
                stacked_output += head_outputs[i];
            }
        }
        if(!concat){
            stacked_output /= head_outputs.size();
        }
        shape(stacked_output,"stacked_output",verbose);

        return stacked_output;
    }
    Eigen::MatrixXd GatLayer::sum_heads(const std::vector<Eigen::MatrixXd>& head_outputs) {
    // Vérifier que le vecteur de sorties de tête n'est pas vide
        if (head_outputs.empty()) {
            throw std::invalid_argument("Les Entete sont vides");
        }
        // Initialiser la matrice de sortie avec la première tête d'attention
        Eigen::MatrixXd summed_output = head_outputs[0];

        // Ajouter les sorties des autres têtes d'attention à la matrice de sortie
        for (int i = 1; i < head_outputs.size(); ++i) {
            summed_output += head_outputs[i];
        }

        return summed_output/head_outputs.size();
    }


    Eigen::MatrixXd GatLayer::forward(const Eigen::MatrixXd& h, const Eigen::MatrixXd& adj,bool isTrain=true) {
        std::vector<Eigen::MatrixXd> head_outputs; // Vecteur pour stocker les sorties de chaque tête
        X=h;
        shape(X,"X",verbose);
        shape(h,"h",verbose);
        cout<<" forward en cours \n"<<endl;
        // Pour chaque tête d'attention, calculer la sortie et l'ajouter au vecteur head_outputs
        for (int i = 0; i < num_of_heads; ++i) {
            cout<<"\t\t forward du head ="<<i+1<<endl;
            head_outputs.push_back(layer[i].forward(h, adj,isTrain)); // Appel à la fonction forward de chaque tête
            cout<<" \t\tfin "<<i+1<<"\n"<<endl;
        }

        // Empiler ou concaténer les sorties des têtes d'attention en fonction de la configuration concat
        Eigen::MatrixXd output;
        if (concat) {
            cout<<" forward en cours (concatenation...)  \n"<<endl;
            output = stack_heads(head_outputs);
            cout<<" forward en cours (concatenation... FIN, utilisation de l'activation RELU)  \n"<<endl;
            return output.unaryExpr([this](double x) { return elu(x,alpha); }); // Concaténation des sorties
        } else {
            cout<<" forward en cours ( pas de concatenation sommation ...)  \n"<<endl;
            output = softmax(sum_heads(head_outputs),1); // Addition des sorties
        }
        return output;
    }

    Eigen::MatrixXd GatLayer::backward(const Eigen::MatrixXd& adj, const Eigen::MatrixXd& gradput, double lr,double beta1 ,double beta2 ,double epsilon){
        std::vector<Eigen::MatrixXd> head_gradients;
        std::vector<Eigen::MatrixXd> grad_output;

        // Pour chaque tête d'attention, calculer le gradient et l'ajouter au vecteur head_gradients
        bool p = false;
        if(out_features*num_of_heads==gradput.cols() and num_of_heads > 1){
           p=true;
           for(int i=0;i< num_of_heads; ++i){
            // nous devons gerer le cas ou nous transitons de deux couche ayant les head de taille differentes
            grad_output.push_back(gradput.block(0,i*out_features,gradput.rows(),out_features));
           }
        }
        for (int i = 0; i < num_of_heads; ++i) {
            Eigen::MatrixXd head_gradient = layer[i].backward_update_parameters( X, adj,
             (p==false)? gradput : grad_output[i]
             ,lr,beta1,beta2,epsilon);
            head_gradients.push_back(head_gradient);
        }
        // Combinez les gradients de chaque tête pour obtenir le gradient total
        Eigen::MatrixXd total_gradient = sum_heads(head_gradients);
        return total_gradient;
    }

    // Versioned deserialization (optional, for future compatibility)
   