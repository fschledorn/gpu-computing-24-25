#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def ex2():
    data_per_threadcount = {}
    for line in open("data/ex2.csv"):
        size, threads, h2d, kernel, d2h = line.split(",")
        size, threads = int(size), int(threads)
        h2d, kernel, d2h = float(h2d), float(kernel), float(d2h)
        if threads not in data_per_threadcount:
            data_per_threadcount[threads] = ([], [], [], [])
        data_per_threadcount[threads][0].append(size)
        data_per_threadcount[threads][1].append(h2d)
        data_per_threadcount[threads][2].append(kernel)
        data_per_threadcount[threads][3].append(d2h)


    fig = plt.figure()
    ax = fig.add_subplot()
    size, h2d, kernel, d2h = data_per_threadcount[1024]
    ax.plot(size, h2d, "o-", label="host to device copy")
    ax.plot(size, kernel, "o-", label="kernel runtime")
    ax.plot(size, d2h, "o-", label="device to host copy")
    ax.plot(size, [a + b + c for a, b, c in zip(h2d, kernel, d2h)], "o-", label="total runtime")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("matrix size")
    ax.set_ylabel("runtime (s)")
    ax.set_title("transfer/kernel/total runtimes for varying matrix sizes")
    fig.savefig("plots/ex2/1024_threads.svg")


    fig = plt.figure()
    ax = fig.add_subplot()
    for tc, (size, h2d, kernel, d2h) in data_per_threadcount.items():
        ax.plot(size, [a + b + c for a, b, c in zip(h2d, kernel, d2h)], "o-", label=f"{tc} threads per block")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("matrix size")
    ax.set_ylabel("total runtime (s)")
    ax.set_title("total runtime for varying threads per block and matrix sizes")
    fig.savefig("plots/ex2/varying_threads.svg")

def ex3():
    threads, _, kernel, _ = np.loadtxt("data/ex3/varying_threads.csv", delimiter=",").T
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(threads, kernel, "o-")
    ax.set_xlabel("threads per block")
    ax.set_ylabel("kernel runtime (s)")
    ax.set_ybound(lower = 0)

    ax.grid()
    ax.set_title("kernel runtime with varying threads per block on a 8192x8192 matrix")
    fig.savefig("plots/ex3/varying_threads.svg")

    size, h2d, kernel, d2h = np.loadtxt("data/ex3/varying_size.csv", delimiter=",").T
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(size, kernel, "o-", label="kernel runtime")
    ax.plot(size, kernel + h2d + d2h, "o-", label="total runtime")
    ax.set_xlabel("threads per block")
    ax.set_ylabel("runtime (s)")
    ax.legend()
    ax.grid()
    ax.set_title("kernel runtime with varying matrix sizes and 1024 threads per block")
    fig.savefig("plots/ex3/varying_size.svg")

def ex4():
    cached_data = np.loadtxt("data/ex4/naive.csv", delimiter=",")
    size, _, _, cached, _ = cached_data[cached_data[:,1] == 1024].T

    uncached_data = np.loadtxt("data/ex2.csv", delimiter=",")
    _, _, _, uncached, _ = uncached_data[uncached_data[:,1] == 1024].T

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(size, cached, "o-", label="L1 enabled")
    ax.plot(size, uncached, "o-", label="L1 disabled")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("matrix size")
    ax.set_ylabel("kernel runtime (s)")
    ax.set_title("Naive matrix multiplication with L1 enabled/disabled")

    fig.savefig("plots/ex4/naive.svg")

    size, _, cached, _ = np.loadtxt("data/ex4/varying_size.csv", delimiter=",").T
    _, _, uncached, _ = np.loadtxt("data/ex3/varying_size.csv", delimiter=",").T

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(size, cached, "o-", label="L1 enabled")
    ax.plot(size, uncached, "o-", label="L1 disabled", lw=0.75, ms=4)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("matrix size")
    ax.set_ylabel("kernel runtime (s)")
    ax.set_title("Tiled matrix multiplication with L1 enabled/disabled")

    fig.savefig("plots/ex4/tiled.svg")

# ex2()
# ex3()
ex4()