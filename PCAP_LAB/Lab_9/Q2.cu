#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


__global__ void transformMatrixRows(float *matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        for (int col = 0; col < cols; col++) {
            float elem = matrix[row * cols + col];
            
            // Apply power transformation with rounding to fix precision errors
            matrix[row * cols + col] = roundf(__powf(elem, row + 1));
        }
    }
}


// Host function to read and transform matrix
void transformMatrix(float *h_matrix, int rows, int cols) {
    size_t matrixSize = rows * cols * sizeof(float);
    float *d_matrix;
    
    // Allocate device memory
    cudaMalloc(&d_matrix, matrixSize);
    
    // Copy matrix to device
    cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);
    
    // Configure grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    transformMatrixRows<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, rows, cols);
    
    // Copy transformed matrix back to host
    cudaMemcpy(h_matrix, d_matrix, matrixSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_matrix);
}

int main() {
    // Example usage
    int rows = 3, cols = 4;
    float matrix[] = {
        1, 2, 3, 4,
        6, 5, 8, 3,
        2, 4, 10, 1
    };
    
    printf("Original Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    
    transformMatrix(matrix, rows, cols);
    
    printf("\nTransformed Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    
    return 0;
}
