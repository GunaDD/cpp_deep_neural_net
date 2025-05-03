main: ./src/main.cpp 
	g++ -std=c++17 ./src/main.cpp -o ./bin/main

clean: 
	rm ./bin/main