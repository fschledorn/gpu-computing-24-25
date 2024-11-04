#include "chTimer.h"
#include <stdio.h>

__device__ int i;

__global__ void busyWait(int cycles, bool b) {
    long long start = clock64();
    long long stop = start + cycles;
    long long current;
    while((current = clock64()) < stop) {
        if (b) i = stop - current;
    }
}

double timeBusyWait(int cycles) {
    const int cIter = 10000;

    double elapsed = 0;
    for (int i = 0; i < cIter; i++) {
        chTimerTimestamp start, stop;
        chTimerGetTime(&start);
        // this measures synchronous time which is not what the exercise asked for.
        // however, the async startup time didn't seem to go up even when looping
        // for millions of cycles, and i don't know why:(
        busyWait<<<1, 1>>>(cycles, false);
        chTimerGetTime(&stop);
        elapsed += chTimerElapsedTime(&start, &stop);
    }
    return elapsed;
}

int main() {
    // launch the kernel once to prevent long latency on the first launch
    timeBusyWait(0);

    double initial = timeBusyWait(0);
    printf("initial: %0.2e\n", initial); fflush(stdout);

    // limit to 10 million cycles as a precaution
    for (int c = 0; c < 10'000'000; c += 100) {
        double t = timeBusyWait(c);
        if (t >= 1.99 * initial) {
            printf("%d cycles: %0.2e\n", c, t); fflush(stdout);
            break;
        }
    }
}
