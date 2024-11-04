/*
 *
 * nullKernelAsync.cu
 *
 * Microbenchmark for throughput of asynchronous kernel launch.
 *
 * Build with: nvcc -I ../chLib <options> nullKernelAsync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include "chTimer.h"

__global__
void
NullKernel()
{
}

double measure_async(int numBlocks, int numThreads) {
    const int cIterations = 10000;
    
    // Measure asynchronous launch time
    //printf( "Measuring asynchronous launch time... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        NullKernel<<<numBlocks, numThreads>>>();
    }
    cudaDeviceSynchronize();
    chTimerGetTime( &stop );

    {
        double microseconds = 1e6 * chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (double) cIterations;

        //printf( "%.2f us\n", usPerLaunch );
        return usPerLaunch;
    }
}

double measure_sync(int numBlocks, int numThreads) {
    const int cIterations = 10000;
    // Measure synchronous launch time
    //printf( "Measuring synchronous launch time... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) {
        NullKernel<<<numBlocks, numThreads>>>();
        cudaDeviceSynchronize();  // Synchronize after each launch
    }
    chTimerGetTime( &stop );

    {
        double microseconds = 1e6 * chTimerElapsedTime( &start, &stop );
        double usPerLaunch = microseconds / (double) cIterations;

        //printf( "%.2f us\n", usPerLaunch );
        return usPerLaunch;
    }
}

int
main()
{   
    for (int numBlocks = 1; numBlocks <= 16384; numBlocks *= 2) {
        for (int numThreads = 1; numThreads <= 1024; numThreads *= 2) {
            double async = measure_async(numBlocks, numThreads);
            double sync = measure_sync(numBlocks, numThreads);
            // Output the results
            printf("%d, %d, %f, %f\n", numBlocks, numThreads, async, sync);
        }
    }
    return 0;
}
