#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void multiplyMatrixElementWise(int *A, int *B, int *C, int m, int n, int p) {
    // C[i][j] = sum(A[i][k] * B[k][j]) for k = 0 to n-1
    // A is m x n, B is n x p, C is m x p
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < p) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

__global__ void multiplyMatrixRowWise(int *A, int *B, int *C, int n, int p) {
    // Each block handles one row of the result matrix
    int row = blockIdx.y;
    
    for (int col = 0; col < p; col++) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

__global__ void multiplyMatrixColWise(int *A, int *B, int *C, int m, int n, int p) {
    // Each block handles one column of the result matrix
    int col = blockIdx.x;
    
    for (int row = 0; row < m; row++) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

void displayMatrix(int *mat, int m, int n) {
    for (int i = 0; i < m; i++) {
        printf("\n");
        for (int j = 0; j < n; j++) {
            printf("%d ", mat[i * n + j]);
        }
    }
    printf("\n");
}

int main() {
    // Host Allocation
    int m, n, p;
    int *matA, *matB, *matC, *matD, *matE;

    printf("Enter size of matrices for multiplication:\n");
    printf("Matrix A (m x n): ");
    scanf("%d %d", &m, &n);
    printf("Matrix B (n x p): ");
    scanf("%d %d", &n, &p);
    
    int sizeA = m * n * sizeof(int);
    int sizeB = n * p * sizeof(int);
    int sizeC = m * p * sizeof(int);
    
    matA = (int*)malloc(sizeA);
    matB = (int*)malloc(sizeB);
    matC = (int*)malloc(sizeC);
    matD = (int*)malloc(sizeC);
    matE = (int*)malloc(sizeC);

    // Random Initialization
    for (int i = 0; i < m * n; i++) {
        matA[i] = rand() % 10;  // Using smaller numbers to avoid overflow
    }
    
    for (int i = 0; i < n * p; i++) {
        matB[i] = rand() % 10;
    }
    
    for (int i = 0; i < m * p; i++) {
        matC[i] = 0;
        matD[i] = 0;
        matE[i] = 0;
    }

    // Display Host Matrices
    printf("\nMatrix A (%d x %d):", m, n);
    displayMatrix(matA, m, n);
    
    printf("\nMatrix B (%d x %d):", n, p);
    displayMatrix(matB, n, p);

    // Device Allocation
    int *d_matA, *d_matB, *d_matC, *d_matD, *d_matE;
    cudaMalloc(&d_matA, sizeA);
    cudaMalloc(&d_matB, sizeB);
    cudaMalloc(&d_matC, sizeC);
    cudaMalloc(&d_matD, sizeC);
    cudaMalloc(&d_matE, sizeC);

    // Host To Device Copy
    cudaMemcpy(d_matA, matA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matC, matC, sizeC, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matD, matD, sizeC, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matE, matE, sizeC, cudaMemcpyHostToDevice);

    // Element Wise Kernel Setup (using 2D grid and blocks)
    dim3 blockSize(16, 16);
    dim3 gridSize((p + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    multiplyMatrixElementWise<<<gridSize, blockSize>>>(d_matA, d_matB, d_matC, m, n, p);
    cudaMemcpy(matC, d_matC, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_matC);
    
    // Row Wise Kernel Setup
    dim3 gridRowWise(1, m, 1);
    dim3 blockRowWise(1, 1, 1);
    multiplyMatrixRowWise<<<gridRowWise, blockRowWise>>>(d_matA, d_matB, d_matD, n, p);
    cudaMemcpy(matD, d_matD, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_matD);

    // Col Wise Kernel Setup
    dim3 gridColWise(p, 1, 1);
    dim3 blockColWise(1, 1, 1);
    multiplyMatrixColWise<<<gridColWise, blockColWise>>>(d_matA, d_matB, d_matE, m, n, p);
    cudaMemcpy(matE, d_matE, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_matE);

    // Device Memory Free
    cudaFree(d_matA);
    cudaFree(d_matB);

    // Display Result
    printf("\nMatrix C (Element Wise) (%d x %d):", m, p);
    displayMatrix(matC, m, p);

    printf("\nMatrix D (Row Wise) (%d x %d):", m, p);
    displayMatrix(matD, m, p);

    printf("\nMatrix E (Col Wise) (%d x %d):", m, p);
    displayMatrix(matE, m, p);

    // Host Memory Free
    free(matA);
    free(matB);
    free(matC);
    free(matD);
    free(matE);

    return 0;
}
