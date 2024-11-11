#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

table = np.loadtxt("run.log", delimiter=',')

size = table[:,0]
hostToDev = table[:,1]
pinnedToDev = table[:,2]
devToHost = table[:,3]
devToPinned = table[:,4]
devToDev = table[:,5]

fig, time_ax = plt.subplots()

time_ax.set_xscale("log")
time_ax.set_xlabel("Memory size (bytes)")
time_ax.set_yscale("log")
time_ax.set_ylabel("Transfer time (s)")

time_ax.plot(size, hostToDev,
             size, pinnedToDev,
             size, devToHost,
             size, devToPinned,
             size, devToDev, marker="o")
time_ax.legend(["host to device", "pinned to device", "device to host", "device to pinned", "device to device"])

plt.grid(True)

fig.savefig("time.svg")


fig, throughput_ax = plt.subplots()

throughput_ax.set_xscale("log")
throughput_ax.set_yscale("log")
throughput_ax.set_xlabel("Memory size (bytes)")
throughput_ax.set_ylabel("Throughput (GiB/s)")

throughput_ax.plot(size, size / hostToDev / (1 << 30),
                   size, size / pinnedToDev / (1 << 30),
                   size, size / devToHost / (1 << 30),
                   size, size / devToPinned / (1 << 30),
                   size, size / devToDev / (1 << 30), marker="o")
throughput_ax.legend(["host to device", "pinned to device", "device to host", "device to pinned", "device to device"])

plt.grid(True)

fig.savefig("throughput.svg")
