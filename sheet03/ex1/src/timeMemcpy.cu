#include "chTimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

template<typename F>
double time(F callback) {
    chTimerTimestamp start, stop;
    chTimerGetTime(&start);
    const int cIter = 10;
    for (int i = 0; i < cIter; i++) {
        callback();
    }
    chTimerGetTime(&stop);
    return chTimerElapsedTime(&start, &stop) / cIter;
}

int main() {
    double hostToDev, devToPinned, pinnedToDev, devToHost, devToDev;

    auto timeSize = [&](int size) {
        void* hostMem = malloc(size);
        void* pinnedMem, *deviceMem, *deviceMem2;
        cudaError_t err = cudaMallocHost(&pinnedMem, size);
        if (err) printf("cudaMallocHost error: %s, size %i\n", cudaGetErrorName(err), size);

        err = cudaMalloc(&deviceMem, size);
        if (err) printf("cudaMalloc error (deviceMem): %s, size %i\n", cudaGetErrorName(err), size);

        err = cudaMalloc(&deviceMem2, size);
        if (err) printf("cudaMalloc error (deviceMem2): %s, size %i\n", cudaGetErrorName(err), size);

        memset(hostMem, 0x11, size);

        hostToDev = time([&](){
            cudaMemcpy(deviceMem, hostMem, size, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        });

        devToPinned = time([&](){
            cudaMemcpy(pinnedMem, deviceMem, size, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        });

        pinnedToDev = time([&](){
            cudaMemcpy(deviceMem, pinnedMem, size, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        });
        devToHost = time([&](){
            cudaMemcpy(hostMem, deviceMem, size, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        });
        devToDev = time([&](){
            cudaMemcpy(deviceMem, deviceMem2, size, cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
        });

        cudaFree(deviceMem);
        cudaFree(deviceMem2);
        cudaFreeHost(pinnedMem);
        free(hostMem);
        cudaDeviceSynchronize();
    };

    // do one cold run to prevent cuda startup latencies in the timings
    timeSize(1 << 10);

    for (int i = 10; i <= 30; i += 2) {
        int size = 1 << i;
        timeSize(size);

        printf("%d,%0.4e,%0.4e,%0.4e,%0.4e,%0.4e\n", size, hostToDev, pinnedToDev, devToHost, devToPinned, devToDev);
        fflush(stdout);
    }
}
