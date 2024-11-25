#include <iostream>
#include <cstdlib> // required for std::atoi
#include "chTimer.hpp"

void matrix_multiply(unsigned int dimension, const int *matA, const int *matB, int *output) {
    for (size_t i = 0; i < dimension; i++) {
        for (size_t j = 0; j < dimension; j++) {
            output[i * dimension + j] = 0;
            for (size_t k = 0; k < dimension; k++) {
                output[i * dimension + j] += matA[i * dimension + k] * matB[k * dimension + j];
            }
        }
    }
}

int main() {

    for (size_t dim = 128; dim < 5096; dim+=128) {

        int *matA = new int[dim * dim];
        int *matB = new int[dim * dim];

        // init matrix values
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                matA[i * dim + j] = i + j;
                matB[i * dim + j] = i * j;
            }
        }

        // allocate memory for output matrix
        int *output = new int[dim * dim];

        ChTimer timer;
        timer.start();

        // call multiply function
        matrix_multiply(dim, matA, matB, output);

        timer.stop();
        
        double t = timer.getTime();

        std::printf("%ld,%g\n", dim, t);

        // deallocate memory
        delete[] matA;
        delete[] matB;
        delete[] output;
    }

    return 0;
}