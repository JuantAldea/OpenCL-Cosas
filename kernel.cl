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
}
