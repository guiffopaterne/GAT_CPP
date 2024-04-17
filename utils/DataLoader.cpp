
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "DataLoader.hpp"

// class DataLoader {
// public:
//     DataLoader(const std::string& adjacency_filename, const std::string& features_filename, const std::string& labels_filename, const char delimit)
//         : adjacency_filename(adjacency_filename), features_filename(features_filename), labels_filename(labels_filename),delimit(delimit) {}

//     Eigen::MatrixXd load_adjacency_matrix() {
//         return load_csv_to_matrix(adjacency_filename);
//     }

//     Eigen::MatrixXd load_features(bool normalize=true) {
//         return load_csv_to_matrix(features_filename);
//     }

//     Eigen::VectorXd load_labels() {
//         // Charge les étiquettes en tant que vecteur
//         return load_csv_to_vector(labels_filename);
//     }

// private:
//     std::string adjacency_filename;
//     std::string features_filename;
//     std::string labels_filename;
//     char delimit=';';

//     Eigen::MatrixXd load_csv_to_matrix(const std::string& filename,bool normalize=false) {
//         std::ifstream file(filename);
//         if (!file.is_open()) {
//             std::cerr << "Error opening file: " << filename << std::endl;
//             // Gérer l'erreur de manière appropriée, par exemple, en lançant une exception
//         }

//         std::vector<std::vector<double>> data;
//         std::string line;
//         while (std::getline(file, line)) {
//             std::istringstream iss(line);
//             std::vector<double> row;
//             double value;
//             while (iss >> value) {
//                 row.push_back(value);
//                 if (iss.peek() == delimit) {
//                     iss.ignore();
//                 }
//             }
//             data.push_back(row);
//         }

//         Eigen::MatrixXd matrix(data.size(), data[0].size());
//         for (int i = 0; i < data.size(); ++i) {
//             for (int j = 0; j < data[0].size(); ++j) {
//                 matrix(i, j) = data[i][j];
//             }
//         }
//         if(normalize){
//             for (int i = 0; i < matrix.cols(); i++){
//                 matrix.col(i).normalize();
//             }  
//         }
//         return matrix;
//     }

//     Eigen::VectorXd load_csv_to_vector(const std::string& filename) {
//         std::ifstream file(filename);
//         if (!file.is_open()) {
//             std::cerr << "Error opening file: " << filename << std::endl;
//             // Gérer l'erreur de manière appropriée, par exemple, en lançant une exception
//         }

//         std::vector<double> data;
//         double value;
//         while (file >> value) {
//             data.push_back(value);
//             if (file.peek() == delimit) {
//                 file.ignore();
//             }
//         }

//         Eigen::VectorXd vector(data.size());
//         for (int i = 0; i < data.size(); ++i) {
//             vector(i) = data[i];
//         }

//         return vector;
//     }
// };


    DataLoader::DataLoader(const std::string& adjacency_filename, const std::string& features_filename, const std::string& labels_filename, const char delimit)
        : adjacency_filename(adjacency_filename), features_filename(features_filename), labels_filename(labels_filename),delimit(delimit) {}

    Eigen::MatrixXd DataLoader::load_adjacency_matrix() {
        return DataLoader::load_csv_to_matrix(adjacency_filename,false);
    }

    Eigen::MatrixXd DataLoader::load_features(bool normalize) {
        return DataLoader::load_csv_to_matrix(features_filename, normalize);
    }

    Eigen::VectorXd DataLoader::load_labels() {
        // Charge les étiquettes en tant que vecteur
        return DataLoader::load_csv_to_vector(labels_filename);
    }

    Eigen::MatrixXd DataLoader::load_csv_to_matrix(const std::string& filename,bool normalize) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            // Gérer l'erreur de manière appropriée, par exemple, en lançant une exception
        }

        std::vector<std::vector<double>> data;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<double> row;
            double value;
            while (iss >> value) {
                row.push_back(value);
                if (iss.peek() == delimit) {
                    iss.ignore();
                }
            }
            data.push_back(row);
        }

        Eigen::MatrixXd matrix(data.size(), data[0].size());
        for (int i = 0; i < data.size(); ++i) {
            for (int j = 0; j < data[0].size(); ++j) {
                matrix(i, j) = data[i][j];
            }
        }
        if(normalize){
            for (int i = 0; i < matrix.cols(); i++){
                matrix.col(i).normalize();
            }  
        }
        return matrix;
    }

    Eigen::VectorXd DataLoader::load_csv_to_vector(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            // Gérer l'erreur de manière appropriée, par exemple, en lançant une exception
        }

        std::vector<double> data;
        double value;
        while (file >> value) {
            data.push_back(value);
            if (file.peek() == delimit) {
                file.ignore();
            }
        }

        Eigen::VectorXd vector(data.size());
        for (int i = 0; i < data.size(); ++i) {
            vector(i) = data[i];
        }

        return vector;
    }