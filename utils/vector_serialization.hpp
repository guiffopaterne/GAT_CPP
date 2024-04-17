// vector_serialization.hpp

#pragma once
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <vector>
#include "GatLayer.hpp"  // Assurez-vous d'inclure le bon fichier d'en-tête pour GatLayer

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive & ar, std::vector<GatLayer>& vector, const unsigned int version) {
    // Sérialisez la taille du vecteur
    size_t vector_size = vector.size();
    ar & vector_size;

    // Sérialisez chaque élément du vecteur
    for (GatLayer& layer : vector) {
        ar & layer;
    }
}

} // namespace serialization
} // namespace boost
