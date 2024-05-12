CXX := g++
CXXFLAGS := -std=c++11 -g -O3 -lpthread -Iutils
LDFLAGS := -lboost_serialization

SOURCES := utils/functions.cpp model/GatUnit.cpp model/GatLayer.cpp model/GatNetwork.cpp utils/DataLoader.cpp main_sequentielle.cpp
HEADERS := utils/functions.hpp utils/vector_serialization.hpp model/GatUnit.hpp model/GatLayer.hpp model/GatNetwork.hpp utils/DataLoader.hpp

OBJECTS := $(SOURCES:.cpp=.o)
EXECUTABLE := gat

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@


# Rules for dependencies
model/GatUnit.o : model/GatUnit.cpp model/GatUnit.hpp  utils/vector_serialization.hpp utils/functions.hpp
model/GatLayer.o: model/GatLayer.cpp model/GatLayer.hpp model/GatUnit.hpp  utils/vector_serialization.hpp utils/functions.hpp
model/GatNetwork.o: utils/GatNetwork.cpp utils/GatNetwork.hpp utils/GatLayer.hpp model/GatUnit.hpp  utils/vector_serialization.hpp utils/functions.hpp
utils/DataLoader.o: utils/DataLoader.cpp utils/DataLoader.hpp
main_sequentielle.o: main_sequentielle.cpp model/GatLayer.hpp model/GatNetwork.hpp utils/DataLoader.hpp utils/functions.hpp


clean:
	rm -f $(OBJECTS) $(EXECUTABLE)