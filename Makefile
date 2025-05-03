main: ./src/main.cpp 
	g++ -std=c++17 ./src/main.cpp -o ./bin/main

inference: ./src/inference.cpp
	g++ -std=c++17 ./src/inference.cpp -o ./bin/inference

data: ./scripts/fetch_mnist.py
	python ./scripts/fetch_mnist.py 

clean: 
	rm ./bin/main ./bin/inference