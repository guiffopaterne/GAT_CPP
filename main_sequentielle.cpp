#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/program_options.hpp>


#include "model/Gatnetwork.hpp"
#include "utils/DataLoader.hpp"
#include "utils/functions.hpp"
#include <vector>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;



int main(int argc, char* argv[]) {

    string dataset_name = "cora";
    int train_size=160;
    int val_size= 640;
    int NUM_CLASSES=7;
    int batch_size = 32;
    int dropout = 0.6;
    int alpha = 0.2;
    double learning_rate = 0.001; // Paramètre d'apprentissage
    double beta1 = 0.9; // Moment d'ordre 1
    double beta2 = 0.999; // Moment d'ordre 2
    double epsilon = 1e-8;
    int epoch = 10;
    int patience = 2;
    int early_stop = 3;
    bool verbose=false;
    int num_of_layers= 2;
    vector<int> num_heads_per_layer = {1,8,1};
    vector<int> num_features_per_layer = {8,8,NUM_CLASSES};

    // Define options description
  po::options_description all_options("Allowed options");
  all_options.add_options()
  ("help","Imprime l'aide")
  ("train_size", po::value<int>(&train_size)->default_value(train_size), "Training data size")
  ("val_size", po::value<int>(&val_size)->default_value(val_size), "Validation data size")
  ("num_classes", po::value<int>(&NUM_CLASSES)->default_value(NUM_CLASSES), "Number of classes")
  ("batch_size", po::value<int>(&batch_size)->default_value(batch_size), "Batch size")
  ("dropout", po::value<int>(&dropout)->default_value(dropout), "Dropout rate")
  ("alpha", po::value<int>(&alpha)->default_value(alpha), "Hyperparameter alpha")
  ("learning_rate", po::value<double>(&learning_rate)->default_value(learning_rate), "Learning rate")
  ("beta1", po::value<double>(&beta1)->default_value(beta1), "Momentum beta1")
  ("beta2", po::value<double>(&beta2)->default_value(beta2), "Momentum beta2")
  ("epsilon", po::value<double>(&epsilon)->default_value(epsilon), "Epsilon value")
  ("epoch", po::value<int>(&epoch)->default_value(epoch), "Number of epochs")
  ("patience", po::value<int>(&patience)->default_value(patience), "Patience for early stopping")
  ("early_stop", po::value<int>(&early_stop)->default_value(early_stop), "Early stopping threshold")
  ("verbose,v", po::bool_switch(&verbose)->default_value(verbose), "Verbose output")
  ("num_layers", po::value<int>(&num_of_layers)->default_value(num_of_layers), "Number of network layers")
  ("num_heads", po::value<std::vector<int>>(&num_heads_per_layer)->default_value(num_heads_per_layer), "Number of heads per layer (comma separated)")
  ("dataset_name", po::value<string>(&dataset_name)->default_value(dataset_name),"Nom du dataset (cora par défaut)");
  // No need for default value for num_features_per_layer, adjust it based on your data

  // Parse command line arguments
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, all_options), vm);
    po::notify(vm);
  } catch(const po::error& e) {
    std::cerr << "Erreur sur les argement consulte l'aide: " << e.what() << std::endl;
    return 1;
  }


   
    GAT gat = GAT(num_of_layers,num_heads_per_layer,num_features_per_layer, NUM_CLASSES, dropout,alpha,verbose);
    const std::string adjacency_filename = "data/" + dataset_name + "/adjacency.csv";
    const std::string features_filename =  "data/" + dataset_name + "/features.csv";
    const std::string labels_filename =    "data/" + dataset_name + "/labels.csv";
    // Créer une instance de DataLoader avec les chemins des fichiers
    DataLoader data_loader(adjacency_filename, features_filename, labels_filename,';');
    // Charger les données depuis les fichiers CSV
    cout<<"chargement de la matrice d'adjence"<<endl;
    Eigen::MatrixXd adjacency_matrix = data_loader.load_adjacency_matrix();
    cout<<"chargement de la matrice de caracteristique"<<endl;
    Eigen::MatrixXd features = data_loader.load_features(true);
    cout<<"chargement de la matrice de labels"<<endl;
    Eigen::VectorXd labels = data_loader.load_labels();

     Eigen::VectorXi train_mask,val_mask,test_mask;
    std::tie(train_mask,val_mask,test_mask)=createMask(labels.size(),train_size,val_size,false);
    int  num_nodes = features.rows();
    int num_features = features.cols();
    // train_batch();
    cout<<"Fin de la Construction du Gat"<<endl;
    cout<<"Conversion du label en onehot encoder"<<endl;
    gat.train(features, adjacency_matrix, labels, train_mask, val_mask,epoch,learning_rate,beta1,beta2,epsilon,patience,early_stop);    

    // Évaluation avec le masque de test (facultatif, dépend de votre configuration)
    gat.evaluate(features, adjacency_matrix, labels, test_mask);
    //  gat.save_model("saved_model.txt");
    // Chargement du modèle
    // GAT loaded_model;
    // loaded_model.load_model("model/saved_model.txt");

    return 0;
}