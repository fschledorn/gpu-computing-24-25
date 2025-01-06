CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I. -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS= -O2 --compiler-options "-O2 -Wall -Wextra"
# NVCC_FLAGS= -Xptxas -flcm=cg -Xptxas -fscm=cg -O2 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/matmul

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/matmul: ./src/matmul.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ ./src/matmul.cu $(INC) $(LIB)
