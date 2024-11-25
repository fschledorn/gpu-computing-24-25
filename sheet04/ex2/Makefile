NVCC=nvcc
CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64

.PHONY: build
build: ./bin/ex2 ./bin/ex3 ./bin/ex4

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/ex2: ./src/ex2.cu
	# $(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
	$(NVCC) -arch=compute_60 -code=sm_70 -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
./bin/ex3: ./src/ex3.cu
	# $(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
	$(NVCC) -arch=compute_60 -code=sm_70 -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
./bin/ex4: ./src/ex4.cpp
	g++ -O3 -Wall -Wextra -o $@ $^ $(INC)
