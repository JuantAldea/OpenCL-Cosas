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

def call_kernel(kernel, parameters, debug = False):
    evt = kernel(*parameters)
    evt.wait()
    if debug: 
        print("T: %f == %s" % (1e-9*(evt.profile.end - evt.profile.start), kernel.function_name))
    return evt.profile.end - evt.profile.start

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

kernels_file = open ("mul.cl", "r")

kernels_src = kernels_file.read()
kernels_file.close()

prg = cl.Program(ctx, kernels_src).build()

print("Work size = %d" % a_np.size)

#mul_kernel_parameters = (queue, (a_np.size,), None, a_g, b_g, r_g, np.int32(identityN.shape[0]), np.int32(identityN.shape[1]), np.int32(identityN.size), global_offset=None, wait_for=None, g_times_l=False)
zero_kernel_parameters = (queue, (a_np.size,), None, r_g,  np.int32(identityN.shape[0]), np.int32(identityN.shape[1]), np.int32(identityN.size));
mul_kernel_parameters = (queue, (a_np.size,), (972, ), a_g, b_g, r_g, np.int32(identityN.shape[0]), np.int32(identityN.shape[1]), np.int32(identityN.size))
mul_kernel_parameters2 = (queue, (a_np.size,), (81, ), a_g, b_g, r_g, np.int32(identityN.shape[0]), np.int32(identityN.shape[1]), np.int32(identityN.size))

time_samples = {}
total_time = {}
kernel_parameters = {}

for kernel in prg.all_kernels():
    time_samples[kernel.function_name] = []
    total_time[kernel.function_name] = 0
    if kernel.function_name == 'zero':
        kernel_parameters[kernel.function_name] = zero_kernel_parameters
    else:
        kernel_parameters[kernel.function_name] = mul_kernel_parameters


excluded_kernels = ['zero']

kernel_list = [kernel.function_name for kernel in prg.all_kernels() if kernel.function_name not in excluded_kernels]
kernel_list.sort()

n_tests = 10000

for i in range(n_tests):
    if i % 100 == 0 and i != 0:
        print("    t      std    n = %d" % (i,))
        for kernel in kernel_list:
            print("%f %f %s" % (1e-9 * (total_time[kernel] / float(i)),1e-9 * np.std(time_samples[kernel]), kernel))

    for kernel in prg.all_kernels():
        time = call_kernel(kernel, kernel_parameters[kernel.function_name])
        total_time[kernel.function_name] += time
        time_samples[kernel.function_name].append(time)


print("    t      std    n = %d" % (i,))
for kernel in kernel_list:
    print("%f %f %s" % (1e-9 * (total_time[kernel] / float(n_tests)),1e-9 * np.std(time_samples[kernel]), kernel))


cl.enqueue_copy(queue, r_np, r_g)

print("A * B =")
print(r_np)

#la comparacion solo tiene sentido si A o B == I
if int(sys.argv[1]) < 2:
    print("||B - A * B|| = %f" % np.linalg.norm(b_np - r_np))

