CXX := g++
CXXFLAGS := -std=c++11 -g -O3 -lpthread -Iutils
LDFLAGS := -lboost_serialization

SOURCES := utils/functions.cpp utils/GatLayer.cpp utils/GatNetwork.cpp utils/DataLoader.cpp main.cpp
HEADERS := utils/functions.hpp utils/vector_serialization.hpp utils/GatLayer.hpp utils/GatNetwork.hpp utils/DataLoader.hpp

OBJECTS := $(SOURCES:.cpp=.o)
EXECUTABLE := gat

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@


# Rules for dependencies
utils/GatLayer.o: utils/GatLayer.cpp utils/GatLayer.hpp utils/vector_serialization.hpp utils/functions.hpp
utils/GatNetwork.o: utils/GatNetwork.cpp utils/GatNetwork.hpp utils/GatLayer.hpp utils/vector_serialization.hpp utils/functions.hpp
utils/DataLoader.o: utils/DataLoader.cpp utils/DataLoader.hpp
main.o: main.cpp utils/GatLayer.hpp utils/GatNetwork.hpp utils/DataLoader.hpp utils/functions.hpp


clean:
	rm -f $(OBJECTS) $(EXECUTABLE)