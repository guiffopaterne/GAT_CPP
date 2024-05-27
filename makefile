CXX := g++
CXXFLAGS := -std=c++17 -g -O3 -lpthread -Iutils
LDFLAGS := -lboost_serialization -lboost_program_options -lboost_filesystem -lboost_system

SOURCES := utils/functions.cpp utils/DataLoader.cpp model/GatUnit.cpp model/GatLayer.cpp model/GatNetwork.cpp training/train.cpp  main_sequentielle.cpp
HEADERS := utils/functions.hpp utils/DataLoader.hpp model/GatUnit.hpp model/GatLayer.hpp model/GatNetwork.hpp training/train.hpp

OBJECTS := $(SOURCES:.cpp=.o)
EXECUTABLE := gat

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LDFLAGS)


# Rules for dependencies
utils/functions.o : utils/functions.cpp utils/functions.hpp
utils/DataLoader.o: utils/DataLoader.cpp utils/DataLoader.hpp
model/GatUnit.o : model/GatUnit.cpp model/GatUnit.hpp utils/functions.hpp
model/GatLayer.o: model/GatLayer.cpp model/GatLayer.hpp model/GatUnit.hpp utils/functions.hpp
model/GatNetwork.o: model/GatNetwork.cpp model/GatNetwork.hpp model/GatLayer.hpp model/GatUnit.hpp utils/functions.hpp
training/train.o: training/train.cpp training/train.hpp  model/GatNetwork.hpp  utils/functions.hpp
main_sequentielle.o: main_sequentielle.cpp training/train.hpp model/GatNetwork.hpp utils/DataLoader.hpp utils/functions.hpp


clean:
	rm -f $(OBJECTS) $(EXECUTABLE)