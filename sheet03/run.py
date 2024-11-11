#!/usr/bin/env python

import subprocess
# import numpy as np


def run_kernel(kernel_args, kernel_flags={}):
    process_args = ["bin/memCpy"]
    for name, val in kernel_args.items():
        process_args.append(name)
        process_args.append(str(val))
    process_args += list(kernel_flags)
    return subprocess.run(process_args, capture_output=True, check=True).stdout.decode()


def write(data, path):
    with open(path, "w") as f:
        f.write(data)


def ex1():
    pageable = "".join(run_kernel({"-s": 1 << i}) for i in range(10, 31, 2))
    pinned = "".join(run_kernel({"-s": 1 << i}, {"-p"}) for i in range(10, 31, 2))
    write(pageable, "data/pageable.csv")
    write(pinned, "data/pinned.csv")


def ex2_1():
    one_block = "".join(
        run_kernel(
            {"-s": 1 << i, "-t": 1 << j, "-g": 1, "-i": 2}, {"--global-coalesced"}
        )
        for j in range(0, 11, 2)
        for i in range(10, 31, 2)
    )
    write(one_block, "data/one_block.csv")

def ex2_2():
    # optimal thread count is 1024
    one_k_threads = "".join(
        run_kernel(
            {"-s": 1 << i, "-t": 1 << 10, "-g": 1 << j, "-i": 20}, {"--global-coalesced"}
        )
        for i in range(10, 31, 2)
        for j in range(max(6, i - 12))
    )
    write(one_k_threads, "data/1K_threads.csv")


def ex3():
    lines = []
    for size_bits in [20, 30]:
        # size = 1 << i
        block_bits = 10
        for stride_bits in range(7):
            # gridDim = size / stride / blockDim / sizeof(int)
            grid_bits = size_bits - stride_bits - block_bits - 2
            lines.append(
                run_kernel(
                    {
                        "-t": 1 << block_bits,
                        "-g": 1 << grid_bits,
                        "--stride": 1 << stride_bits,
                    },
                    {"--global-stride"},
                )
            )

    write("".join(lines), "data/strided.csv")

    # print(run_kernel({"-t": 1 << 10, "-g": 1 << 10, "--stride": 1 << 5, "-i": 1000}, {"--global-stride"}))


def ex4():
    lines = []
    for size_bits in [20, 30]:
        # size = 1 << i
        block_bits = 10
        for offset_bits in range(7):
            # gridDim = size / stride / blockDim / sizeof(int)
            grid_bits = size_bits - block_bits - 2
            lines.append(
                run_kernel(
                    {
                        "-t": 1 << block_bits,
                        "-g": 1 << grid_bits,
                        "--offset": 1 << offset_bits,
                    },
                    {"--global-offset"},
                )
            )

    write("".join(lines), "data/offset.csv")


# ex1()
# ex2()
ex2_2()
# ex3()
# ex4()
