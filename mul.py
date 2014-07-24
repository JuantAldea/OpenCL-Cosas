#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import sys

def roundUpToNextPowerOfTwo(x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x += 1
    return x;

if len(sys.argv) != 5:
    print("%s [B_identidad = 0, B_rand = 1, ambos_tres_rand = _] [dimension] [n_matrices] [padd = 1; !padd = _]" % sys.argv[0]);
    exit(0)

dimension = int(sys.argv[2]);
n_matrices = int(sys.argv[3]);

identityN = np.identity(dimension, dtype=np.float32)
a_np = None

if sys.argv[1] == "0":
    a_np = np.vstack(tuple(identityN for i in range(n_matrices)))
    b_np = np.vstack(tuple(identityN for i in range(n_matrices)))
elif sys.argv[1] == "1":
    a_np = np.vstack(tuple(identityN for i in range(n_matrices)))
    b_np = np.random.random(a_np.shape).astype(np.float32)
else:
    a_np = np.random.random((dimension, dimension * n_matrices)).astype(np.float32)
    b_np = np.random.random(a_np.shape).astype(np.float32)

if sys.argv[4] == '1':
    next_power_of_two = roundUpToNextPowerOfTwo(a_np.size)
    print("len = %d; padded = %d" % (a_np.size, next_power_of_two))
    a_np = a_np.reshape(a_np.size);
    b_np = b_np.reshape(b_np.size);
    a_np = np.hstack((a_np, np.zeros(next_power_of_two - a_np.size)))
    b_np = np.hstack((b_np, np.zeros(next_power_of_two - b_np.size)))

access_counter = np.zeros(a_np.shape, dtype=np.int);

r_np = np.zeros_like(a_np)
r2_np = np.zeros_like(a_np)

print("A =")
print(a_np)
print()
print("B =")
print(b_np)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags

a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
r_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
r2_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

source2 = """
kernel void zero(global float *C, const int dim1, const int dim2, const int length)
{
   int j = get_global_id(0);
   int matrix_index =  j / length;
   int i = j % dim1;
   j = (j - matrix_index * length) / dim1;
   C[matrix_index * length + i * dim1 + j] = 0;
}
"""

source = """
kernel void mul(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int length)
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

//   C[matrix_index * length + i * dim1 + j] = matrix_index * length + i * dim1 + j;
//   C[matrix_index * length + i * dim1 + j] = (j %3  == 0) * get_local_size(0) + (j%3 == 1) * get_num_groups(0) + (j%3 == 2) * get_global_size(0);
//   C[matrix_index * length + i * dim1 + j] = C[matrix_index * length + i * dim1 + j] + 1 ;
}
"""

source3 = """
kernel void mul_local(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int length)
{
   local float a[1024];
   local float b[1024];

   int j = get_global_id(0);
   int matrix_index =  j / length;
   int i = j % dim1;
   j = (j - matrix_index * length) / dim1;
    
   event_t evt = async_work_group_copy(a, &(A[matrix_index * length]), get_local_size(0), 0);
   wait_group_events(1, &evt);
   event_t evt2 = async_work_group_copy(b, &(B[matrix_index * length]), get_local_size(0), 0);
   wait_group_events(1, &evt2);

   float sum = 0;
   for (int k = 0; k < dim1; k++) {
      sum += a[i * dim1 + k] * b[k * dim2 + j];
   }

   C[matrix_index * length + i * dim1 + j] = sum;

//   C[matrix_index * length + i * dim1 + j] = matrix_index * length + i * dim1 + j;
//   C[matrix_index * length + i * dim1 + j] = (j %3  == 0) * get_local_size(0) + (j%3 == 1) * get_num_groups(0) + (j%3 == 2) * get_global_size(0);
//   C[matrix_index * length + i * dim1 + j] = C[matrix_index * length + i * dim1 + j] + 1 ;
}
"""

prg = cl.Program(ctx, source + source2 + source3).build()
print("Work size = %d" % a_np.size)
prg.zero(queue, (a_np.size,), None, r_g,  np.int32(identityN.shape[0]), np.int32(identityN.shape[1]), np.int32(identityN.size), global_offset=None, wait_for=None, g_times_l=False).wait()

exec_evt_2 = prg.mul(queue, (a_np.size,), None, b_g, a_g, r2_g, np.int32(identityN.shape[0]), np.int32(identityN.shape[1]), np.int32(identityN.size), global_offset=None, wait_for=None, g_times_l=False)
exec_evt_2.wait()
print("T2: %f" % (1e-9*(exec_evt_2.profile.end - exec_evt_2.profile.start)))

exec_evt_1 = prg.mul(queue, (a_np.size,), None, a_g, b_g, r_g, np.int32(identityN.shape[0]), np.int32(identityN.shape[1]), np.int32(identityN.size), global_offset=None, wait_for=None, g_times_l=False)
exec_evt_1.wait()
print("T1: %f" % (1e-9*(exec_evt_1.profile.end - exec_evt_1.profile.start)))



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

