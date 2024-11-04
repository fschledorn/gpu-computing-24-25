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
    // printf("memSize,hostToDev,pinnedToDev,devToHost,devToPinned\n"); fflush(stdout);
    
    double hostToDev, devToPinned, pinnedToDev, devToHost;

    auto timeSize = [&](int size) {
        void* hostMem = malloc(size);
        void* pinnedMem, *deviceMem;
        cudaError_t err = cudaMallocHost(&pinnedMem, size);
        if (err) printf("%s\n", cudaGetErrorName(err));
        err = cudaMalloc(&deviceMem, size);
        if (err) printf("%s\n", cudaGetErrorName(err));
        memset(hostMem, 0x11, size);
        hostToDev = time([&](){ cudaMemcpy(deviceMem, hostMem, size, cudaMemcpyHostToDevice); });
        devToPinned = time([&](){ cudaMemcpy(pinnedMem, deviceMem, size, cudaMemcpyDeviceToHost); });
        pinnedToDev = time([&](){ cudaMemcpy(deviceMem, pinnedMem, size, cudaMemcpyHostToDevice); });
        devToHost = time([&](){ cudaMemcpy(hostMem, deviceMem, size, cudaMemcpyDeviceToHost); });

        cudaFree(deviceMem);
        cudaFreeHost(pinnedMem);
        free(hostMem);
    };

    // do one cold run to prevent cuda startup latencies in the timings
    timeSize(1 << 10);

    for (int i = 10; i <= 30; i += 2) {
        int size = 1 << i;
        timeSize(size);

        printf("%d,%0.4e,%0.4e,%0.4e,%0.4e\n", size, hostToDev, pinnedToDev, devToHost, devToPinned); fflush(stdout);
    }
}