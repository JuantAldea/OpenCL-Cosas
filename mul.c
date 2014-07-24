
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

const char *get_error_string(cl_int err)
{
    switch (err) {
    case 0:
        return "CL_SUCCESS";
    case -1:
        return "CL_DEVICE_NOT_FOUND";
    case -2:
        return "CL_DEVICE_NOT_AVAILABLE";
    case -3:
        return "CL_COMPILER_NOT_AVAILABLE";
    case -4:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:
        return "CL_OUT_OF_RESOURCES";
    case -6:
        return "CL_OUT_OF_HOST_MEMORY";
    case -7:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8:
        return "CL_MEM_COPY_OVERLAP";
    case -9:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case -10:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11:
        return "CL_BUILD_PROGRAM_FAILURE";
    case -12:
        return "CL_MAP_FAILURE";

    case -30:
        return "CL_INVALID_VALUE";
    case -31:
        return "CL_INVALID_DEVICE_TYPE";
    case -32:
        return "CL_INVALID_PLATFORM";
    case -33:
        return "CL_INVALID_DEVICE";
    case -34:
        return "CL_INVALID_CONTEXT";
    case -35:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case -36:
        return "CL_INVALID_COMMAND_QUEUE";
    case -37:
        return "CL_INVALID_HOST_PTR";
    case -38:
        return "CL_INVALID_MEM_OBJECT";
    case -39:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40:
        return "CL_INVALID_IMAGE_SIZE";
    case -41:
        return "CL_INVALID_SAMPLER";
    case -42:
        return "CL_INVALID_BINARY";
    case -43:
        return "CL_INVALID_BUILD_OPTIONS";
    case -44:
        return "CL_INVALID_PROGRAM";
    case -45:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46:
        return "CL_INVALID_KERNEL_NAME";
    case -47:
        return "CL_INVALID_KERNEL_DEFINITION";
    case -48:
        return "CL_INVALID_KERNEL";
    case -49:
        return "CL_INVALID_ARG_INDEX";
    case -50:
        return "CL_INVALID_ARG_VALUE";
    case -51:
        return "CL_INVALID_ARG_SIZE";
    case -52:
        return "CL_INVALID_KERNEL_ARGS";
    case -53:
        return "CL_INVALID_WORK_DIMENSION";
    case -54:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case -55:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case -56:
        return "CL_INVALID_GLOBAL_OFFSET";
    case -57:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case -58:
        return "CL_INVALID_EVENT";
    case -59:
        return "CL_INVALID_OPERATION";
    case -60:
        return "CL_INVALID_GL_OBJECT";
    case -61:
        return "CL_INVALID_BUFFER_SIZE";
    case -62:
        return "CL_INVALID_MIP_LEVEL";
    case -63:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    default:
        return "Unknown OpenCL error";
    }
}

char *read_kernel(const char *path)
{
    FILE *fd = fopen(path, "r");
    fseek(fd, 0, SEEK_END);
    size_t kernel_size = ftell(fd);
    rewind(fd);
    char *kernel_src;
    kernel_src = malloc(sizeof(char) * (kernel_size + 1));
    kernel_src[kernel_size] = '\0';
    fread(kernel_src, sizeof(char), kernel_size, fd);
    fclose(fd);
    return kernel_src;
}

int main(int argc, char **argv)
{
    unsigned int dim1 = 3;
    unsigned int dim2 = 3;
    unsigned int n_matrices = 1;
    unsigned int DATA_SIZE = dim1 * dim2 * n_matrices;

    float data[DATA_SIZE];	// original data set given to device
    float results[DATA_SIZE];	// results returned from device

    int err;			// error code returned from api calls
    unsigned int correct;	// number of correct results returned

    size_t global;		// global domain size for our calculation
    size_t local;		// local domain size for our calculation

    cl_device_id device_id;	// compute device id
    cl_context context;		// compute context
    cl_command_queue commands;	// compute command queue
    cl_program program;		// compute program
    cl_kernel kernel;		// compute kernel

    cl_mem a;			// device memory used for the input array
    cl_mem b;			// device memory used for the input array
    cl_mem c;			// device memory used for the output array


    int i;
    for (i = 0; i < DATA_SIZE; i++) {
        data[i] = rand() / (float) RAND_MAX;
    }
    // Connect to a compute device

    cl_uint numPlatforms;	//the NO. of platforms
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    scanf("%d", &err);
    cl_platform_id *platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    err = clGetDeviceIDs(platforms[2], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        printf("%s\n", get_error_string(err));
        return EXIT_FAILURE;
    }

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    char *kernel_src = read_kernel("kernel.cl");
    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_src, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    
    kernel = clCreateKernel(program, "mul", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        printf("%s\n", get_error_string(err));
        exit(1);
    }
    
    a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * DATA_SIZE, NULL, NULL);
    b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * DATA_SIZE, NULL, NULL);
    c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE, NULL, NULL);

    err =
        clEnqueueWriteBuffer(commands, a, CL_TRUE, 0,
                             sizeof(float) * DATA_SIZE, data, 0, NULL, NULL);
    err |=
        clEnqueueWriteBuffer(commands, b, CL_TRUE, 0,
                             sizeof(float) * DATA_SIZE, data, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    // Set the arguments to our compute kernel
    //

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &dim1);
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &dim2);
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &DATA_SIZE);

    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    err =
        clGetKernelWorkGroupInfo(kernel, device_id,
                                 CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    printf("GROUP_SIZE %d\n", local);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
    
    global = DATA_SIZE;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err) {
        printf("%s\n", get_error_string(err));
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    
    clFinish(commands);

    err =
        clEnqueueReadBuffer(commands, c, CL_TRUE, 0,
                            sizeof(float) * DATA_SIZE, results, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    // Shutdown and cleanup
    //
    clReleaseMemObject(a);
    clReleaseMemObject(b);
    clReleaseMemObject(c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    free(kernel_src);
    return 0;
}
