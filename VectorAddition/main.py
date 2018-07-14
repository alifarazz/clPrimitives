#!/usr/bin/env python3
# -*-coding:utf-8-*-
import pyopencl as cl
import numpy as np

MAXN = 256


def main():
    a_np = np.array([i for i in range(10, MAXN + 20)]).astype(np.float32)
    b_np = np.array([i for i in range(MAXN)]).astype(np.float32)
    c_np = np.empty_like(b_np)

    ctx = cl.create_some_context(answers=["0"])
    # ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    c_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=c_np)
    # cl.enqueue_read_buffer(queue, c_dev, c)

    prg = cl.Program(ctx, open("adder.cl").read()).build()

    print("ok")

    # prg.add(queue,
    #         a_np.shape,
    #         (a_np.nbytes, b_np.nbytes, c_np.nbytes), a_g, b_g, c_g)
    print(a_np.size)
    prg.vecadd(queue, a_np.shape, (a_np.size, ), a_g, b_g, c_g)

    cl.enqueue_read_buffer(queue, c_g, c_np).wait()

    print(a_np, b_np, c_np, sep='\n')

    return None


if __name__ == "__main__":
    main()
