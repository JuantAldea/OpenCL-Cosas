kernel void zero(global float *C, const int dim1, const int dim2, const int matrix_length)
{
    int j = get_global_id(0);
    int matrix_index =  j / matrix_length;
    int i = j % dim1;
    j = (j - matrix_index * matrix_length) / dim1;
    C[matrix_index * matrix_length + i * dim1 + j] = 0;
}

kernel void mul_flat_global(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int matrix_length)
{
    int j = get_global_id(0);
    int matrix_index =  j / matrix_length;
    int i = j % dim1;
    j = (j - matrix_index * matrix_length) / dim1;
    int matrix_base_index = matrix_index * matrix_length;
    int A_and_C_constant_index = matrix_base_index + i * dim1; 
    int B_constant_index = matrix_base_index + j;

    float sum = 0;
    for (int k = 0; k < dim1; k++) {
        sum += A[A_and_C_constant_index + k] * B[B_constant_index + k * dim2];
    }

    C[A_and_C_constant_index + j] = sum;
}

kernel void mul_flat_local(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int matrix_length)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    // only if workgroup == matrix_length
    // uint matrix_base_index = global_id - local_id;
    // uint row = local_id / dim1;
    // uint col = local_id % dim1;
    uint matrix_base_index = (local_id / matrix_length) * matrix_length;
    uint matrix_local_index = local_id - matrix_base_index;
    uint row = matrix_local_index / dim1;
    uint col = matrix_local_index % dim1;
    uint a_constant_index = matrix_base_index + row * dim1;
    uint b_constant_index = matrix_base_index + col; 
    
    local float a[972], b[972];
   
    a[local_id] = A[global_id];
    b[local_id] = B[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    for (int k = 0; k < dim1; k++) {
        sum += a[a_constant_index + k] * b[b_constant_index + dim1 * k];
    }

    C[global_id] = sum;
}

kernel void mul_flat_local_unrolled(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int matrix_length)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    
    uint matrix_base_index = (local_id / matrix_length) * matrix_length;
    uint matrix_local_index = local_id - matrix_base_index;
    uint row = matrix_local_index / dim1;
    uint col = matrix_local_index % dim1;
    uint a_constant_index = matrix_base_index + row * dim1;
    uint b_constant_index = matrix_base_index + col; 
    local float a[1024], b[1024];
   
    a[local_id] = A[global_id];
    b[local_id] = B[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum;
    sum  = a[a_constant_index + 0] * b[b_constant_index + dim1 * 0];
    sum += a[a_constant_index + 1] * b[b_constant_index + dim1 * 1];
    sum += a[a_constant_index + 2] * b[b_constant_index + dim1 * 2];
    sum += a[a_constant_index + 3] * b[b_constant_index + dim1 * 3];
    sum += a[a_constant_index + 4] * b[b_constant_index + dim1 * 4];
    sum += a[a_constant_index + 5] * b[b_constant_index + dim1 * 5];
    sum += a[a_constant_index + 6] * b[b_constant_index + dim1 * 6];
    sum += a[a_constant_index + 7] * b[b_constant_index + dim1 * 7];
    sum += a[a_constant_index + 8] * b[b_constant_index + dim1 * 8];

    C[global_id] = sum;
}

kernel void mul_flat_local_unrolled_vectors3(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int matrix_length)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    
    uint matrix_base_index = (local_id / matrix_length) * matrix_length;
    uint matrix_local_index = local_id - matrix_base_index;
    uint row = matrix_local_index / dim1;
    uint col = matrix_local_index % dim1;
    uint a_constant_index = matrix_base_index + row * dim1;
    uint b_constant_index = matrix_base_index + col; 
    local float a[1024], b[1024];
   
    a[local_id] = A[global_id];
    b[local_id] = B[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    float3 a_v;
    float3 b_v;
    float3 c_v;

    a_v.x = a[a_constant_index + 0];
    a_v.y = a[a_constant_index + 1];
    a_v.z = a[a_constant_index + 2];
    b_v.x = b[b_constant_index + dim1 * 0];
    b_v.y = b[b_constant_index + dim1 * 1];
    b_v.z = b[b_constant_index + dim1 * 2];
    c_v = a_v * b_v;
    sum += c_v.x + c_v.y + c_v.z;

    a_v.x = a[a_constant_index + 3];
    a_v.y = a[a_constant_index + 4];
    a_v.z = a[a_constant_index + 5];
    b_v.x = b[b_constant_index + dim1 * 3];
    b_v.y = b[b_constant_index + dim1 * 4];
    b_v.z = b[b_constant_index + dim1 * 5];
    c_v = a_v * b_v;
    sum += c_v.x + c_v.y + c_v.z;
    
    a_v.x = a[a_constant_index + 6];
    a_v.y = a[a_constant_index + 7];
    a_v.z = a[a_constant_index + 8];
    b_v.x = b[b_constant_index + dim1 * 6];
    b_v.y = b[b_constant_index + dim1 * 7];
    b_v.z = b[b_constant_index + dim1 * 8];
    c_v = a_v * b_v;
    sum += c_v.x + c_v.y + c_v.z;

    C[global_id] = sum;
}

kernel void mul_flat_local_unrolled_vectors4(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int matrix_length)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    
    uint matrix_base_index = (local_id / matrix_length) * matrix_length;
    uint matrix_local_index = local_id - matrix_base_index;
    uint row = matrix_local_index / dim1;
    uint col = matrix_local_index % dim1;
    uint a_constant_index = matrix_base_index + row * dim1;
    uint b_constant_index = matrix_base_index + col; 
    local float a[1024], b[1024];
   
    a[local_id] = A[global_id];
    b[local_id] = B[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    float4 a_v;
    float4 b_v;
    float4 c_v;

    a_v.s0 = a[a_constant_index + 0];
    a_v.s1 = a[a_constant_index + 1];
    a_v.s2 = a[a_constant_index + 2];
    a_v.s3 = a[a_constant_index + 3];
    b_v.s0 = b[b_constant_index + dim1 * 0];
    b_v.s1 = b[b_constant_index + dim1 * 1];
    b_v.s2 = b[b_constant_index + dim1 * 2];
    b_v.s3 = b[b_constant_index + dim1 * 3];
    c_v = a_v * b_v;
    sum += c_v.s0 + c_v.s1 + c_v.s2 + c_v.s3;

    a_v.s0 = a[a_constant_index + 4];
    a_v.s1 = a[a_constant_index + 5];
    a_v.s2 = a[a_constant_index + 6];
    a_v.s3 = a[a_constant_index + 7];
    b_v.s0 = b[b_constant_index + dim1 * 4];
    b_v.s1 = b[b_constant_index + dim1 * 5];
    b_v.s2 = b[b_constant_index + dim1 * 6];
    b_v.s3 = b[b_constant_index + dim1 * 7];
    c_v = a_v * b_v;
    sum += c_v.s0 + c_v.s1 + c_v.s2 + c_v.s3 + a[a_constant_index + 8] * b[b_constant_index + dim1 * 8];

    C[global_id] = sum;
}

kernel void mul_flat_local_unrolled_vectors4_dot(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int matrix_length)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    
    uint matrix_base_index = (local_id / matrix_length) * matrix_length;
    uint matrix_local_index = local_id - matrix_base_index;
    uint row = matrix_local_index / dim1;
    uint col = matrix_local_index % dim1;
    uint a_constant_index = matrix_base_index + row * dim1;
    uint b_constant_index = matrix_base_index + col; 
    local float a[1024], b[1024];
   
    a[local_id] = A[global_id];
    b[local_id] = B[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    float4 a_v;
    float4 b_v;
    float4 c_v;

    a_v.s0 = a[a_constant_index + 0];
    a_v.s1 = a[a_constant_index + 1];
    a_v.s2 = a[a_constant_index + 2];
    a_v.s3 = a[a_constant_index + 3];
    b_v.s0 = b[b_constant_index + dim1 * 0];
    b_v.s1 = b[b_constant_index + dim1 * 1];
    b_v.s2 = b[b_constant_index + dim1 * 2];
    b_v.s3 = b[b_constant_index + dim1 * 3];
    
    sum += dot(a_v, b_v);

    a_v.s0 = a[a_constant_index + 4];
    a_v.s1 = a[a_constant_index + 5];
    a_v.s2 = a[a_constant_index + 6];
    a_v.s3 = a[a_constant_index + 7];
    b_v.s0 = b[b_constant_index + dim1 * 4];
    b_v.s1 = b[b_constant_index + dim1 * 5];
    b_v.s2 = b[b_constant_index + dim1 * 6];
    b_v.s3 = b[b_constant_index + dim1 * 7];
    
    sum += dot(a_v, b_v) + a[a_constant_index + 8] * b[b_constant_index + dim1 * 8];

    C[global_id] = sum;
}

kernel void mul_flat_local_unrolled_vectors8(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int matrix_length)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    
    uint matrix_base_index = (local_id / matrix_length) * matrix_length;
    uint matrix_local_index = local_id - matrix_base_index;
    uint row = matrix_local_index / dim1;
    uint col = matrix_local_index % dim1;
    uint a_constant_index = matrix_base_index + row * dim1;
    uint b_constant_index = matrix_base_index + col; 
    local float a[1024], b[1024];
   
    a[local_id] = A[global_id];
    b[local_id] = B[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    float8 a_v;
    float8 b_v;
    float8 c_v;
    a_v.s0 = a[a_constant_index + 0];
    a_v.s1 = a[a_constant_index + 1];
    a_v.s2 = a[a_constant_index + 2];
    a_v.s3 = a[a_constant_index + 3];
    a_v.s4 = a[a_constant_index + 4];
    a_v.s5 = a[a_constant_index + 5];
    a_v.s6 = a[a_constant_index + 6];
    a_v.s7 = a[a_constant_index + 7];

    b_v.s0 = b[b_constant_index + dim1 * 0];
    b_v.s1 = b[b_constant_index + dim1 * 1];
    b_v.s2 = b[b_constant_index + dim1 * 2];
    b_v.s3 = b[b_constant_index + dim1 * 3];
    b_v.s4 = b[b_constant_index + dim1 * 4];
    b_v.s5 = b[b_constant_index + dim1 * 5];
    b_v.s6 = b[b_constant_index + dim1 * 6];
    b_v.s7 = b[b_constant_index + dim1 * 7];

    c_v = a_v * b_v;
    sum += c_v.s0 + c_v.s1 + c_v.s2 + c_v.s3
        + c_v.s4 + c_v.s5 + c_v.s6 + c_v.s7
        + a[a_constant_index + 8] * b[b_constant_index + dim1 * 8];

    C[global_id] = sum;
}

kernel void mul_flat_local_unrolled_vectors16(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int matrix_length)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    
    uint matrix_base_index = (local_id / matrix_length) * matrix_length;
    uint matrix_local_index = local_id - matrix_base_index;
    uint row = matrix_local_index / dim1;
    uint col = matrix_local_index % dim1;
    uint a_constant_index = matrix_base_index + row * dim1;
    uint b_constant_index = matrix_base_index + col; 
    local float a[1024], b[1024];
   
    a[local_id] = A[global_id];
    b[local_id] = B[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    float16 a_v;
    float16 b_v;
    float16 c_v;
    a_v.s0 = a[a_constant_index + 0];
    a_v.s1 = a[a_constant_index + 1];
    a_v.s2 = a[a_constant_index + 2];
    a_v.s3 = a[a_constant_index + 3];
    a_v.s4 = a[a_constant_index + 4];
    a_v.s5 = a[a_constant_index + 5];
    a_v.s6 = a[a_constant_index + 6];
    a_v.s7 = a[a_constant_index + 7];
    a_v.s8 = a[a_constant_index + 8];

    b_v.s0 = b[b_constant_index + dim1 * 0];
    b_v.s1 = b[b_constant_index + dim1 * 1];
    b_v.s2 = b[b_constant_index + dim1 * 2];
    b_v.s3 = b[b_constant_index + dim1 * 3];
    b_v.s4 = b[b_constant_index + dim1 * 4];
    b_v.s5 = b[b_constant_index + dim1 * 5];
    b_v.s6 = b[b_constant_index + dim1 * 6];
    b_v.s7 = b[b_constant_index + dim1 * 7];
    b_v.s8 = b[b_constant_index + dim1 * 8];

    c_v = a_v * b_v;
    sum += c_v.s0 + c_v.s1 + c_v.s2 + c_v.s3
        + c_v.s4 + c_v.s5 + c_v.s6 + c_v.s7 + c_v.s7;

    C[global_id] = sum;
}

kernel void mul_flat_local_unrolled_vectors16_vload(global float *A, global float *B, global float *C, const int dim1, const int dim2, const int matrix_length)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    
    uint matrix_base_index = (local_id / matrix_length) * matrix_length;
    uint matrix_local_index = local_id - matrix_base_index;
    uint row = matrix_local_index / dim1;
    uint col = matrix_local_index % dim1;
    uint a_constant_index = matrix_base_index + row * dim1;
    uint b_constant_index = matrix_base_index + col; 
    local float a[1024], b[1024];
   
    a[local_id] = A[global_id];
    b[local_id] = B[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    float16 a_v;
    float16 b_v;
    float16 c_v;
    a_v = vload16(0, &a[a_constant_index]);
    
    b_v.s0 = b[b_constant_index + dim1 * 0];
    b_v.s1 = b[b_constant_index + dim1 * 1];
    b_v.s2 = b[b_constant_index + dim1 * 2];
    b_v.s3 = b[b_constant_index + dim1 * 3];
    b_v.s4 = b[b_constant_index + dim1 * 4];
    b_v.s5 = b[b_constant_index + dim1 * 5];
    b_v.s6 = b[b_constant_index + dim1 * 6];
    b_v.s7 = b[b_constant_index + dim1 * 7];
    b_v.s8 = b[b_constant_index + dim1 * 8];

    c_v = a_v * b_v;
    sum += c_v.s0 + c_v.s1 + c_v.s2 + c_v.s3
        + c_v.s4 + c_v.s5 + c_v.s6 + c_v.s7 + c_v.s7;

    C[global_id] = sum;
}
