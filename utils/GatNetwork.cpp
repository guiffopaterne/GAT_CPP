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
#include "GatLayer.hpp"
#include "GatNetwork.hpp"
#include "vector_serialization.hpp" 
#include "functions.hpp"

using namespace std;


GAT::GAT(int nfeat, int nhid, int nclass, double dropout, double alpha)
    : dropout(dropout), nhid(nhid),alpha(alpha), nclass(nclass) {
    attentions.push_back(GatLayer(nfeat, 8, dropout, alpha, true));
    for (int i = 1; i < nhid; ++i) {
        attentions.push_back(GatLayer(8, 8, dropout, alpha, true));
    }
    out_att = GatLayer(8 , nclass, dropout, alpha, false);
}

Eigen::MatrixXd GAT::forward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj){
    // X = X.unaryExpr([this](double x) { return dropNode(x); });
    int el = 1;
    int numAttentions = attentions.size();
    std::vector<Eigen::MatrixXd> attentionResults;

    Eigen::MatrixXd r = attentions[0].forward(X, adj);
    for (int i = 1; i < nhid; ++i) {
        cout<<" Layer n-"<< i+1<<endl;
        r = attentions[i].forward(r, adj);
        // attentionResults.push_back(r);
        el++;
    }
    r = r.unaryExpr([this](double x) { return dropNode(x); });
    r = out_att.forward(r, adj);
    r = r.unaryExpr([this](double x) { return elu(x); });
    return softmax(r,1);

    // Concatenate along dimension 1 (column-wise)
    // Eigen::MatrixXd h = Eigen::MatrixXd::Zero(X.rows(), numAttentions * this->nhid);
    // cout<<"les dimensions des h ="<<h.rows()<<"x"<<h.cols()<<endl;
    // int i =0;
    // for(const auto& att : attentionResults){
    //     cout<<"les dimensions des h concat  h(0,"<<i * att.cols()<<","<<att.rows()<<","<<att.cols()<<")"<<endl;
    //     h.block(0, i * att.cols(), att.rows(), att.cols()) = att;
    //     i+=1;
    // }
    // h = h.unaryExpr([this](double x) { return dropNode(x); });
    // cout<<" Layer n-"<< el<<endl;
    // h = out_att.forward(h, adj);
    // h = h.unaryExpr([this](double x) { return elu(x); });
    // return softmax(h,1); // Log softmax
    }

    Eigen::MatrixXd GAT::backward(const Eigen::MatrixXd& X, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output) {
    // Backward pass for GAT
    Eigen::MatrixXd grad_h = grad_output; 
    // Gradients with respect to the output of the final layer
    // Backward pass through the last layer
    grad_h = std::get<2>(out_att.backward(attentions.back().forward(X, adj), adj, grad_h));

    // Backward pass through each attention layer
    for (int i = nhid; i > 0; --i) {
        grad_h = std::get<2>(attentions[i].backward(X, adj, grad_h));
        // Eigen::MatrixXd grad_h = std::get<2>(attentions[i].backward(X, adj, grad_h));
        // grad_h = grad_h + grad_x; // Summing gradients from all heads
        // grad_h = grad_h + grad_x; // Summing gradients from all heads
    }
    return grad_h;
}
Eigen::VectorXd GAT::evaluate_node(const Eigen::MatrixXd& node_features, const Eigen::MatrixXd& adj) {
    // Forward pass through the GAT for a single node
    Eigen::MatrixXd h = node_features.transpose();
    // h = h.unaryExpr([this](double x) { return dropNode(x); });
    h = attentions[nhid].forward(h, adj);
    for (int i = 1; i < nhid; ++i) {
            h= attentions[i].forward(h, adj);
        }
    h = out_att.forward(h, adj);
    h = h.unaryExpr([this](double x) { return dropNode(x); });
    return h.unaryExpr([](double x) { return std::log(x) / std::log(2.71828); }); // Log softmax
}


void GAT::train(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
                const Eigen::MatrixXd& labels, const Eigen::VectorXd& train_mask, 
                const Eigen::VectorXd& val_mask, int num_epochs) {
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Matrices pour accumuler les gradients sur l'ensemble du lot
        Eigen::MatrixXd batch_grad_x = Eigen::MatrixXd::Zero(train_mask.size(), features.cols());
        Eigen::MatrixXd batch_grad_adj = Eigen::MatrixXd::Zero(train_mask.size(), adjacency_matrix.cols());
        Eigen::MatrixXd batch_grad_output = Eigen::MatrixXd::Zero(train_mask.size(),1);
        int j = 0;
        // Eigen::MatrixXd X = features.unaryExpr([this](double x) { return dropNode(x); });

        Eigen::MatrixXd h =  forward(features,adjacency_matrix);
        h = logSoftmax(h,1);
        Eigen::MatrixXd grad_output = (h - labels).transpose();        
        double acc = accuracy(h,labels,train_mask);
        cout <<"Accurancy "<<acc<<endl;
        Eigen::MatrixXd out =  backward(features,adjacency_matrix,grad_output);
// mise a jour des poids

        // Évaluation du modèle sur l'ensemble de validation
    double val_accuracy = evaluate(features, adjacency_matrix, labels, val_mask);
    std::cout << "Epoch " << epoch + 1<<"Validation Accuracy: " << val_accuracy << std::endl;

            }
}


void GAT::train_batch(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
                const Eigen::MatrixXd& labels, const Eigen::VectorXd& train_mask, 
                const Eigen::VectorXd& val_mask, int batch_size, int num_epochs) {
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (int batch_start = 0; batch_start < features.rows(); batch_start += batch_size) {
        int b = batch_start + batch_size;
        int r = features.rows();
        int batch_end = std::min(b, r);
        cout<<"batch debut "<<b<<" fin"<<batch_end<<endl;

        // Matrices pour accumuler les gradients sur l'ensemble du lot
        Eigen::MatrixXd batch_grad_x = Eigen::MatrixXd::Zero(batch_size, features.cols());
        Eigen::MatrixXd batch_grad_adj = Eigen::MatrixXd::Zero(batch_size, adjacency_matrix.cols());
        Eigen::MatrixXd batch_grad_output = Eigen::MatrixXd::Zero(batch_size,1);

        // Boucle sur le lot
        for (int i = batch_start; i < batch_end; ++i) {
            Eigen::VectorXd node_features = features.row(i);
            Eigen::VectorXd node_labels(nclass); // Supposons une classification binaire
            node_labels =labels.row(i);
            Eigen::MatrixXd node_features_col = node_features.transpose();
            Eigen::VectorXd predictions = evaluate_node(node_features_col, adjacency_matrix);
            cout<<"predictions de la loss "<<predictions.tail(5)<<endl;
            Eigen::VectorXd prediction_log = predictions.array().log();
            double loss = -node_labels.dot(prediction_log);
            cout<<"calculer de la loss "<<loss<<endl;

            // Calculer les gradients pour cet exemple individuel
            Eigen::MatrixXd grad_x, grad_adj;
            Eigen::MatrixXd grad_output;
            grad_x = grad_adj = grad_output = (predictions - node_labels).transpose();

            // Accumuler les gradients sur l'ensemble du lot
            cout<<"Calculer des gradients "<<predictions.tail(5)<<endl;
            batch_grad_x.row(i - batch_start) = grad_x;
            batch_grad_adj.row(i - batch_start) = grad_adj;
            // batch_grad_output(i - batch_start) = grad_output;
            batch_grad_output(i - batch_start) = grad_output.mean();

            std::cout << "Epoch " << epoch + 1 << ", Example " << i + 1 << ", Loss: " << loss << std::endl;
        }

        // Mise à jour des paramètres du modèle (par exemple, utiliser un optimiseur comme Adam)
        backward_batch(batch_grad_x, batch_grad_adj, batch_grad_output);

        // Évaluation du modèle sur l'ensemble de validation
        double val_accuracy = evaluate(features, adjacency_matrix, labels, val_mask);
        std::cout << "Validation Accuracy: " << val_accuracy << std::endl;
    }
}
}

void GAT::backward_batch(const Eigen::MatrixXd& x, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& grad_output) {
// Backward pass for GAT with batch processing
Eigen::MatrixXd grad_h = grad_output;
 // Gradients with respect to the output of the final layer

// Backward pass through the last layer
Eigen::MatrixXd grad_h_out = std::get<2>(out_att.backward(attentions.back().forward(x, adj), adj, grad_h));

// Backward pass through each attention layer
for (int i = nhid - 1; i >= 0; --i) {
    // Calculate average gradients over the batch for each attention layer
    Eigen::MatrixXd avg_grad_x = std::get<2>(attentions[i].backward(x, adj, grad_h)).colwise().mean();
    grad_h = grad_h + avg_grad_x; // Summing gradients from all heads
}

// Update parameters using the averaged gradients
backward_batch_update_parameters(grad_h_out, adj, x);
}

// Function to update parameters with averaged gradients for the batch
void GAT::backward_batch_update_parameters(const Eigen::MatrixXd& grad_h_out, const Eigen::MatrixXd& adj, const Eigen::MatrixXd& x) {
// Update parameters for the last layer using the averaged gradient
out_att.backward_update_parameters(x, grad_h_out);

// Update parameters for each attention layer using the averaged gradient
for (int i = nhid - 1; i >= 0; --i) {
    attentions[i].backward_update_parameters(x, grad_h_out);
}
}

// Fonction d'évaluation sur l'ensemble de validation
double GAT::evaluate(const Eigen::MatrixXd& features, const Eigen::MatrixXd& adjacency_matrix,
                const Eigen::MatrixXd& labels, const Eigen::VectorXd& eval_mask) {
    int correct_predictions = 0;
    int total_examples = 0;

    for (int i = 0; i < features.rows(); ++i) {
        if (eval_mask(i) > 0) {
            Eigen::VectorXd node_features = features.row(i);
            Eigen::VectorXd node_labels(2); // Supposons une classification binaire
            node_labels = labels.row(i);
            Eigen::VectorXd predictions = evaluate_node(node_features, adjacency_matrix);
            // Sélectionnez l'indice de la classe prédite
            int predicted_class = (predictions(1) > 0.5) ? 1 : 0;

            // Vérifiez si la prédiction est correcte
            if (predicted_class == static_cast<int>(labels(i))) {
                correct_predictions++;
            }

            total_examples++;
        }
    }

    // Calcul de la précision
    double accuracy = static_cast<double>(correct_predictions) / total_examples;
    return accuracy;
}

void GAT::set_num_epochs(int num_epochs){
    num_epochs=num_epochs;
}


double GAT::dropNode(double x) {
    return (std::rand() / static_cast<double>(RAND_MAX)) > dropout ? x / (1 - dropout) : 0.0;
}

double GAT::elu(double x) {
        return x > 0 ? x : alpha * (std::exp(x) - 1);
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
