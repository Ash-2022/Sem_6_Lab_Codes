#include <cuda_runtime.h>
#include <stdio.h>

// Kernel for sparse matrix-vector multiplication
__global__ void sparseMatrixVectorMul(
    const int *rowPtr,
    const int *colIdx,
    const float *values,
    const float *vector,
    float *result,
    int numRows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows)
    {
        float dotProduct = 0.0f;
        int startIdx = rowPtr[row];
        int endIdx = rowPtr[row + 1];

        for (int j = startIdx; j < endIdx; j++)
        {
            dotProduct += values[j] * vector[colIdx[j]];
        }

        result[row] = dotProduct;
    }
}

// Host function to perform sparse matrix-vector multiplication
void sparseMVMul(
    const int *h_rowPtr,
    const int *h_colIdx,
    const float *h_values,
    const float *h_vector,
    float *h_result,
    int numRows,
    int numNonZero)
{
    // Device memory allocation
    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_vector, *d_result;

    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, numNonZero * sizeof(int));
    cudaMalloc(&d_values, numNonZero * sizeof(float));
    cudaMalloc(&d_vector, numRows * sizeof(float));
    cudaMalloc(&d_result, numRows * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, numNonZero * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, numNonZero * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, numRows * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    sparseMatrixVectorMul<<<blocksPerGrid, threadsPerBlock>>>(
        d_rowPtr, d_colIdx, d_values, d_vector, d_result, numRows);

    // Copy result back to host
    cudaMemcpy(h_result, d_result, numRows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_vector);
    cudaFree(d_result);
}

int main()
{
    // Example usage
    int numRows = 4;
    int numNonZero = 6;

    // CSR format representation
    int h_rowPtr[] = {0, 2, 4, 5, 6};                  // Row pointers
    int h_colIdx[] = {0, 1, 1, 2, 2, 3};               // Column indices
    float h_values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // Non-zero values
    float h_vector[] = {1.0, 2.0, 3.0, 4.0};           // Input vector
    float h_result[4];

    sparseMVMul(h_rowPtr, h_colIdx, h_values, h_vector, h_result, numRows, numNonZero);

    // Print results
    printf("Result of Sparse Matrix-Vector Multiplication:\n");
    for (int i = 0; i < numRows; i++)
    {
        printf("%f ", h_result[i]);
    }
    printf("\n");

    return 0;
}
