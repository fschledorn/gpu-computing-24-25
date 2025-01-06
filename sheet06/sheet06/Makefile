CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I. -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS= -O2 --compiler-options "-O2 -Wall -Wextra"
# NVCC_FLAGS= -arch=compute_60 -code=sm_70 -O2 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/reduce

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/reduce: ./src/reduce.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ ./src/reduce.cu $(INC) $(LIB)
