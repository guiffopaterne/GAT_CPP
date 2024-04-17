#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

class DataLoader {
public:
    DataLoader(const std::string& adjacency_filename, const std::string& features_filename, const std::string& labels_filename, const char delimit);

    Eigen::MatrixXd load_adjacency_matrix();

    Eigen::MatrixXd load_features(bool normalize);

    Eigen::VectorXd load_labels();

private:
    std::string adjacency_filename;
    std::string features_filename;
    std::string labels_filename;
    char delimit=';';

    Eigen::MatrixXd load_csv_to_matrix(const std::string& filename,bool normalize);
    Eigen::VectorXd load_csv_to_vector(const std::string& filename);
};
#endif