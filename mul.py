#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import sys

# a_np = np.random.random((3, 6)).astype(np.float32)
# b_np = np.random.random((3, 6)).astype(np.float32)
# r_np = np.random.random((3, 6)).astype(np.float32)

if len(sys.argv) != 4:
    print("%s [B_identidad = 0, B_rand = 1, ambos_tres_rand = _] [dimension] [n_matrices]" % sys.argv[0]);
    exit(0)

identityN = np.identity(int(sys.argv[2]), dtype=np.float32)

a_np = None

if sys.argv[1] == "0":
    a_np = np.vstack(tuple(identityN for i in range(int(sys.argv[3]))))
    b_np = np.vstack(tuple(identityN for i in range(int(sys.argv[3]))))
elif sys.argv[1] == "1":
    a_np = np.vstack(tuple(identityN for i in range(int(sys.argv[3]))))
    b_np = np.random.random(a_np.shape).astype(np.float32)
else:
    a_np = np.vstack(tuple(np.random.random(identityN.shape).astype(np.float32) for i in range(int(sys.argv[3]))))
    b_np = np.random.random(a_np.shape).astype(np.float32)

r_np = np.zeros_like(a_np)
r2_np = np.zeros_like(a_np)

print("A =")
print(a_np)
print()
print("B =")
print(b_np)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
r_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
r2_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

source = """
__kernel void mul(__global float *A, __global float *B, __global float *C, const int dim1, const int dim2, const int length)
{
   int j = get_global_id(0);
   int matrix_index =  j / length;
   int i = j % dim1;
   j = (j - matrix_index * length) / dim1;
   float sum = 0;
   for (int k = 0; k < dim1; k++) {
      sum += A[matrix_index * length + i * dim1 + k] * B[matrix_index * length + k * dim2 + j];
   }
 
   C[matrix_index * length + i * dim1 + j] = sum;
//   C[matrix_index * length + i * dim1 + j] = matrix_index * legnth + i * dim1 + j;
}
"""

prg = cl.Program(ctx, source).build()

prg.mul(queue, (a_np.size,), None, a_g, b_g, r_g,  np.int32(identityN.shape[0]), np.int32(identityN.shape[1]), np.int32(identityN.size), global_offset=None, wait_for=None, g_times_l=False).wait()
prg.mul(queue, (a_np.size,), None, b_g, a_g, r2_g, np.int32(identityN.shape[0]), np.int32(identityN.shape[1]), np.int32(identityN.size), global_offset=None, wait_for=None, g_times_l=False).wait()

cl.enqueue_copy(queue, r_np, r_g)
cl.enqueue_copy(queue, r2_np, r2_g)

print("A * B =")
print(r_np)
print("B * A =")
print(r2_np)

#la comparacion solo tiene sentido si A == I
if int(sys.argv[1]) < 2:
    print("||B - A * B|| = %f" % np.linalg.norm(b_np - r_np))
    print("||B - B * A|| = %f" % np.linalg.norm(b_np - r2_np))
    print("||A * B - B * A|| = %f" % np.linalg.norm(r_np - r2_np))

