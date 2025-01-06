CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I. -I$(CUDA_ROOT)/include -I/usr/include/SDL2
LIB=-L$(CUDA_ROOT)/lib64 -lhdf5 -lGL -lGLU -lglfw -lGLEW -lcudart -lcuda -lstdc++ -lSDL2
NVCC=nvcc

.PHONY: build
build: ./bin/nbody
build: ./bin/yee
build: ./bin/sdl
build: ./bin/soa
build: ./bin/stream
build: ./bin/streaming

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/nbody: ./src/main.cu
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)

./bin/yee: ./prototyping/yee.cu
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)

./bin/sdl: ./src/main_sdl.cu
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)

./bin/soa: ./src/main_soa.cu
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)

./bin/stream: ./src/main_stream.cu
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)

./bin/streaming: ./src/main_streaming.cu
	$(NVCC) -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)