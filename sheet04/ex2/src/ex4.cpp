#include <algorithm>
#include <cstdio>
#include <memory>
#include <utility>

#include "chTimer.hpp"

// void transpose(float M[], int n) {
//     for (int i = 0; i < n; i++) {
//         for (int j = i + 1; j < n; j++) {
//             std::swap(M[i * n + j], M[j * n + i]);
//         }
//     }
// }

// computes A*B and stores the result in C.
// all matrices are assumed to be n*n and stored
// in a row major layout.
void matmul(float A[], float B[], float C[], int n) {
    // transpose(B, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
                // sum += A[i * n + k] * B[k + n * j];
            }
            C[i * n + j] = sum;
        }
    }
    // transpose(B, n);
}

double time_matmul(int n, int iterations = 1000, bool print = false) {
    auto A = std::make_unique<float[]>(n * n);
    auto B = std::make_unique<float[]>(n * n);
    auto C = std::make_unique<float[]>(n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = i + j;
            B[i * n + j] = i * j;
        }
    }

    ChTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        matmul(A.get(), B.get(), C.get(), n);
    }
    timer.stop();

    if (print) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::printf("%g ", C[i * n + j]);
            }
            std::printf("\n");
        }
    }

    return timer.getTime(iterations);
}

int main() {
    for (int i = 128; i <= 2048 + 512; i += 128) {
        double t = time_matmul(i, std::max(1, 2048 / i));
        std::printf("%d,%g\n", i, t);
    }
}