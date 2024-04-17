#include <iostream>
#include <eigen3/Eigen/Dense>
#include "utils/Gatnetwork.hpp"
#include "utils/DataLoader.hpp"
#include "utils/functions.hpp"
#include <vector>

using namespace std;
using namespace Eigen;

Eigen::VectorXd createMask(int size, double percentage,Eigen::VectorXd &mask) {
        // Crée un masque binaire avec le pourcentage spécifié d'indices à 1
        int num_selected = static_cast<int>(size * percentage);
        for (int i = 0; i < num_selected; ++i) {
            int index;
            do {
                index = rand() % size;
            } while (mask[index] == 1); // Assurez-vous de ne pas sélectionner le même index plusieurs fois
            mask[index] = 1;
        }
        return mask;
    }

int main() {
    // Example usage
    // -I/path/to/eigen
    const std::string adjacency_filename = "data/adjacency.csv";
    const std::string features_filename = "data/features.csv";
    const std::string labels_filename = "data/labels.csv";

    // Créer une instance de DataLoader avec les chemins des fichiers
    DataLoader data_loader(adjacency_filename, features_filename, labels_filename,';');

    // Charger les données depuis les fichiers CSV
    cout<<"chargement de la matrice d'adjence"<<endl;
    Eigen::MatrixXd adjacency_matrix = data_loader.load_adjacency_matrix();
    cout<<"chargement de la matrice de caracteristique"<<endl;
    Eigen::MatrixXd features = data_loader.load_features(true);
    cout<<"chargement de la matrice de labels"<<endl;
    Eigen::VectorXd labels = data_loader.load_labels();
    int  num_nodes = features.rows();
    int num_features = features.cols();
    cout<<"chargement contruction mask"<<endl;
    Eigen::VectorXd mask = Eigen::VectorXd::Zero(num_nodes);
    cout<<"chargement contruction train_mask"<<endl;
    Eigen::VectorXd train_mask = createMask(num_nodes, 0.7,mask);
    cout<<"chargement contruction val_mask"<<endl;
    Eigen::VectorXd val_mask = createMask(num_nodes, 0.15, mask);
    cout<<"chargement contruction test_mask"<<endl;
    Eigen::VectorXd test_mask = createMask(num_nodes, 0.15, mask);
    cout<<"Construction du Gat en cours"<<endl;
    GAT gat = GAT(num_features, 10, 2, 0.6,0.2, 3);
    // train_batch();
    cout<<"Fin de la Construction du Gat"<<endl;
    cout<<"Conversion du label en onehot encoder"<<endl;
    Eigen::MatrixXd labels_onehot = encode_onehot(labels);
    gat.train(features, adjacency_matrix, labels_onehot, train_mask, val_mask,10);

    // Validation avec le masque de validation (facultatif, dépend de votre configuration)
    

    // Évaluation avec le masque de test (facultatif, dépend de votre configuration)
    double test_accuracy = gat.evaluate(features, adjacency_matrix, labels, test_mask);
    std::cout << "Test Accuracy: " << test_accuracy << std::endl;

    //  gat.save_model("saved_model.txt");

    // Chargement du modèle
    // GAT loaded_model;
    // loaded_model.load_model("model/saved_model.txt");

    return 0;
}