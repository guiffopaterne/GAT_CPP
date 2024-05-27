#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

// passager d'argument lors de l'excecution du gat
#include <boost/program_options.hpp>

// Incluez avant Cereal Serialization
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include "utils/DataLoader.hpp"
#include "utils/functions.hpp"
#include "model/Gatnetwork.hpp"
#include "training/train.hpp"


using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

int MAX_HEAD = 10;
int MIN_HEAD = 1;
int MAX_THREAD = 32;

int MAX_NUM_FEATURES = 1000;
int MIN_NUM_FEATURES = 3;



int main(int argc, char* argv[]) {

    string dataset_name = "cora";
    int train_size=160;
    int val_size= 640;
    int test_size=500;
    int NUM_CLASSES=7;
    int batch_size = 32;
    double dropout = 0.6;
    double alpha = 0.2;
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
    vector<int> num_features_per_layer = {1433,8,NUM_CLASSES};
    int continue_training = 0;

    // Define options description
  po::options_description generic("Options Generiques");
  po::options_description data("Options des Donnees");
  po::options_description model("Contruction du Model");
  generic.add_options()
  ("help","Imprime l'aide")
  ("verbose,v", po::bool_switch(&verbose)->default_value(verbose), "Verbose output");
  data.add_options()
  ("train_size,t", po::value<int>(&train_size)->default_value(train_size), "Training data size")
  ("val_size,v", po::value<int>(&val_size)->default_value(val_size), "Validation data size")
  ("test_size,T", po::value<int>(&test_size)->default_value(test_size), "Test data size")
  ("batch_size,b", po::value<int>(&batch_size)->default_value(batch_size), "Batch size")
  ("dataset_name,d", po::value<string>(&dataset_name)->default_value(dataset_name),"Nom du dataset (cora par défaut)")
  ("num_classes,c", po::value<int>(&NUM_CLASSES)->default_value(NUM_CLASSES), "Number of classes");
  model.add_options()
  ("continue,C", po::value<int>(&continue_training)->default_value(continue_training), "Continue l'entrainement s'il trouve un model donc le nombre d'epoque n'est pas encore atteint")
  ("dropout ,D", po::value<double>(&dropout)->default_value(dropout), "Dropout rate")
  ("alpha,A", po::value<double>(&alpha)->default_value(alpha), "Hyperparameter alpha")
  ("learning_rate, lr", po::value<double>(&learning_rate)->default_value(learning_rate), "Learning rate")
  ("beta1,b1", po::value<double>(&beta1)->default_value(beta1), "Momentum beta1")
  ("beta2.b2", po::value<double>(&beta2)->default_value(beta2), "Momentum beta2")
  ("epsilon,E", po::value<double>(&epsilon)->default_value(epsilon), "Epsilon value")
  ("epoch,e", po::value<int>(&epoch)->default_value(epoch), "Number of epochs")
  ("patience,p", po::value<int>(&patience)->default_value(patience), "Patience for early stopping")
  ("early_stop,es", po::value<int>(&early_stop)->default_value(early_stop), "Early stopping threshold")
  ("num_layers,NL", po::value<int>(&num_of_layers)->default_value(num_of_layers), "Number of network layers")
  ("num_heads,NL", 
    po::value<std::vector<int>>()->multitoken()->notifier([&num_heads_per_layer](std::vector<int> const& raw) {
              for (auto el : raw) {
                  if (el < MIN_HEAD || el > MAX_HEAD)
                      throw std::range_error("num heads per layer:" + std::to_string(el));
                  num_heads_per_layer.push_back(static_cast<int>(el));
              }
          }), "Number of heads per layer (comma separated)")
  ("num_features_per_layer,NF",
    po::value<std::vector<int>>()->multitoken()->notifier([&num_features_per_layer](std::vector<int> const& raw) {
              for (auto el : raw) {
                  if (el < MIN_NUM_FEATURES || el > MAX_NUM_FEATURES)
                      throw std::range_error("num FEATURE  per layer:" + std::to_string(el));
                  num_features_per_layer.push_back(static_cast<int>(el));
              }
          })
          , "Number of features per layer (comma separated)");
  
  // Parse command line arguments
  po::variables_map vm;
  try {
    po::options_description cmdline_options;
    cmdline_options.add(generic).add(data).add(model);
    po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    po::notify(vm);
    if (vm.count("help")) {
      std::cout << cmdline_options << "\n";
      return 1;
    }
    if(vm.count("num_features_per_layer")){

    }
  } catch(const po::error& e) {
    std::cerr << "Erreur sur les argement consulte l'aide: " << e.what() << std::endl;
    return 1;
  }
   
    
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
    cout<<"nombre d'exemple "<<labels.size()<<endl;
    std::tie(train_mask,val_mask,test_mask)=createMask(labels.size(),train_size,val_size,false);
    int  num_nodes = features.rows();
    int num_features = features.cols();
    


    cout<<"Fin de la Construction du Gat"<<endl;
    cout<<"Conversion du label en onehot encoder"<<endl;
    TRAIN t = TRAIN(num_of_layers,num_heads_per_layer,num_features_per_layer, NUM_CLASSES, dropout,alpha,verbose,epoch,learning_rate,beta1,beta2,epsilon,patience,early_stop);
    // cherche s'il existe une model dans le fichier pretrained verifie sil le num_epoch est == a celle donne par l'utilisateur si < continue l'entrainement
    t.excecute(features, adjacency_matrix, labels, train_mask, val_mask);
    // sauvergarde du model
    t.save_model(epoch);
    // Évaluation avec le masque de test (facultatif, dépend de votre configuration)
    t.evaluate(features, adjacency_matrix, labels, test_mask);
    //

    return 0;
}