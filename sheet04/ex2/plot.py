#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

def ex2():

	# g2s
	g2s = np.loadtxt("data/global_to_shmem.csv", delimiter=",")

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.grid(True)
	# ax.set_xticks(np.arange(0, 2049, 256))
	ax.xaxis.set_major_locator(MultipleLocator(8192))
	ax.xaxis.set_minor_locator(AutoMinorLocator(4))

	ax.grid(which="major", linestyle="-")
	ax.grid(which="minor", linestyle="-")

	ax.set_xlabel("mem size")
	# ax.set_xscale("log")
	ax.set_ylabel("bandwidth (GB/s)")
	# ax.set_yscale("log")

	for i in range(1, 7):
		plt.plot(g2s[:,0], g2s[:,i], "-", label=f"{4**(i-1)} threads", marker='o')
	ax.legend()
	fig.savefig("plots/ex2_g2s.svg")

	# s2g
	s2g = np.loadtxt("data/shmem_to_global.csv", delimiter=",")

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.grid(True)
	# ax.set_xticks(np.arange(0, 2049, 256))
	ax.xaxis.set_major_locator(MultipleLocator(8192))
	ax.xaxis.set_minor_locator(AutoMinorLocator(4))

	ax.grid(which="major", linestyle="-")
	ax.grid(which="minor", linestyle="-")

	ax.set_xlabel("mem size")
	# ax.set_xscale("log")
	ax.set_ylabel("bandwidth (GB/s)")
	# ax.set_yscale("log")

	for i in range(1, 7):
		plt.plot(s2g[:,0], s2g[:,i], "-", label=f"{4**(i-1)} threads", marker='o')
	ax.legend()
	fig.savefig("plots/ex2_s2g.svg")


	# g2s variable grid size
	g2sVar = np.loadtxt("data/global_to_shmem_variable_gridsize.csv", delimiter=",")

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.grid(True)

	ax.grid(which="major", linestyle="-")
	ax.grid(which="minor", linestyle="-")

	ax.set_xlabel("grid size")
	# ax.set_xscale("log")
	ax.set_ylabel("bandwidth (GB/s)")
	# ax.set_yscale("log")

	plt.plot(g2sVar[:,0], g2sVar[:,1], "-", label=f"1024 threads", marker='o')
	ax.legend()
	fig.savefig("plots/ex2_g2s_variable_gridsize.svg")


	# s2g variable grid size
	s2gVar = np.loadtxt("data/shmem_to_global_variable_gridsize.csv", delimiter=",")

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.grid(True)

	ax.grid(which="major", linestyle="-")
	ax.grid(which="minor", linestyle="-")

	ax.set_xlabel("grid size")
	# ax.set_xscale("log")
	ax.set_ylabel("bandwidth (GB/s)")
	# ax.set_yscale("log")

	plt.plot(s2gVar[:,0], s2gVar[:,1], "-", label=f"1024 threads", marker='o')
	ax.legend()
	fig.savefig("plots/ex2_s2g_variable_gridsize.svg")

	# shared to register
	s2r = np.loadtxt("data/shmem_to_reg.csv", delimiter=",")

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.grid(True)
	# ax.set_xticks(np.arange(0, 2049, 256))
	ax.xaxis.set_major_locator(MultipleLocator(8192))
	ax.xaxis.set_minor_locator(AutoMinorLocator(4))

	ax.grid(which="major", linestyle="-")
	ax.grid(which="minor", linestyle="-")

	ax.set_xlabel("mem size")
	# ax.set_xscale("log")
	ax.set_ylabel("bandwidth (GB/s)")
	# ax.set_yscale("log")

	for i in range(1, 7):
		plt.plot(s2r[:,0], s2r[:,i], "-", label=f"{4**(i-1)} threads", marker='o')
	ax.legend()
	fig.savefig("plots/ex2_s2r.svg")

	# register to shared
	r2s = np.loadtxt("data/reg_to_shmem.csv", delimiter=",")

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.grid(True)
	# ax.set_xticks(np.arange(0, 2049, 256))
	ax.xaxis.set_major_locator(MultipleLocator(8192))
	ax.xaxis.set_minor_locator(AutoMinorLocator(4))

	ax.grid(which="major", linestyle="-")
	ax.grid(which="minor", linestyle="-")

	ax.set_xlabel("mem size")
	# ax.set_xscale("log")
	ax.set_ylabel("bandwidth (GB/s)")
	# ax.set_yscale("log")

	for i in range(1, 7):
		plt.plot(r2s[:,0], r2s[:,i], "-", label=f"{4**(i-1)} threads", marker='o')
	ax.legend()
	fig.savefig("plots/ex2_r2s.svg")

def ex4():
	data = np.loadtxt("data/matmul.csv", delimiter=",")
	# data = np.loadtxt("data/matmul_transposed.csv", delimiter=",")

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.grid(True)
	# ax.set_xticks(np.arange(0, 2049, 256))
	ax.xaxis.set_major_locator(MultipleLocator(256))
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))

	ax.grid(which="major", linestyle="-")
	ax.grid(which="minor", linestyle="-")

	ax.set_xlabel("N")
	# ax.set_xscale("log")
	ax.set_ylabel("NxN matmul runtime (s)")
	# ax.set_yscale("log")

	ax.plot(data[:,0], data[:,1], 'o-')
	fig.savefig("plots/ex4.svg")

ex2()
