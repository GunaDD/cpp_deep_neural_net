main: ./src/main.cpp 
	g++ -std=c++17 ./src/main.cpp -o ./bin/main

inference: ./src/inference.cpp
	g++ -std=c++17 ./src/inference.cpp -o ./bin/inference

clean: 
	rm ./bin/main ./bin/inference