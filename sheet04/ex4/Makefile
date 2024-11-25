NVCC=nvcc
CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64

.PHONY: build
build: ./bin/ex4

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/ex4: ./src/ex4.cpp
	g++ -O3 -Wall -Wextra -o $@ $^ $(INC)
