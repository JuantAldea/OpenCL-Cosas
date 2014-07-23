#!/usr/bin/env python

import numpy as np
import pyopencl as cl

a_np = np.random.random((3, 3)).astype(np.float32)
b_np = np.random.random((3, 3)).astype(np.float32)
r_np = np.empty_like(a_np)

print(a_np)
print()
print(b_np)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
r_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

source = """
__kernel void mul(__global float *A, __global float *B, __global float *C, const int dim1, const int dim2)
{
   int j = get_global_id(0);
   int matrix_index = j / 9;
   int i = j / 3;
   j = j % 3;
   float sum = 0;
   for (int k = 0; k < dim1; k++) {
      sum += A[matrix_index * 9 + i * dim1 + k] * B[matrix_index * 9 + k * dim2 + j];
   }
 
   C[matrix_index * 9 + i * dim1 + j] = sum;
}
"""

prg = cl.Program(ctx, source).build()

prg.mul(queue, (9,), None, a_g, b_g, r_g, np.int32(3), np.int32(3), global_offset=None, wait_for=None, g_times_l=False).wait()

cl.enqueue_copy(queue, r_np, r_g)


print(r_np)
print(np.linalg.norm(r_np))
