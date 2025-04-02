#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

const int MAX_BLK_THREADS = 1024;

__global__ void addMatrixElementWise(int * A , int * B , int * C , int r , int c){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < (r*c)){
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void addMatrixRowWise(int * A , int * B , int * C){
    int r = blockIdx.y;
    int c = blockDim.x;
    for(int i = 0; i < c; i++){
        C[r*c + i] = A[r*c + i] + B[r*c + i];
    }
}

__global__ void addMatrixColWise(int * A , int * B , int * C , int m , int n){
    int r = threadIdx.x; 
    int c = blockIdx.x; 
    if (r < m) {
        C[r * n + c] = A[r * n + c] + B[r * n + c]; 
    }
}

void displayMatrix(int * mat , int m , int n){
    for(int i = 0; i < m*n; i++){
        if(!(i % m)) printf("\n");
        printf("%d " , mat[i]);
    }
    printf("\n");
}

int main(){
    // Host Allocation
    int m , n;
    int *matA , *matB , *matC , *matD , *matE;

    printf("Enter Size of matrix : (Rows , Col) : ");
    scanf("%d %d" , &m , &n);
    int matSize = m * n * sizeof(int);
    
    matA = (int*) malloc(matSize);
    matB = (int*) malloc(matSize);
    matC = (int*) malloc(matSize);
    matD = (int*) malloc(matSize);
    matE = (int*) malloc(matSize);

    // Rand Initialization
    for(int i = 0; i < m * n;i++){
        matA[i] = rand() % 100;
        matB[i] = rand() % 100;
        matC[i] = 0;
        matD[i] = 0;
        matE[i] = 0;
    }

    // Display Host Matricies
    printf("\nMatrix A :");
    displayMatrix(matA , m , n);
    
    printf("\nMatrix B :");
    displayMatrix(matB , m , n);

    // Device Allocation
    int *d_matA , *d_matB , *d_matC , *d_matD , *d_matE;
    cudaMalloc(&d_matA , matSize);
    cudaMalloc(&d_matB , matSize);
    cudaMalloc(&d_matC , matSize);
    cudaMalloc(&d_matD , matSize);
    cudaMalloc(&d_matE , matSize);

    // Host To Device Copy
    cudaMemcpy(d_matA , matA , matSize , cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB , matB , matSize , cudaMemcpyHostToDevice);
    cudaMemcpy(d_matC , matC , matSize , cudaMemcpyHostToDevice);
    cudaMemcpy(d_matD , matD , matSize , cudaMemcpyHostToDevice);
    cudaMemcpy(d_matE , matE , matSize , cudaMemcpyHostToDevice);

    // Element Wise Kernal Setup
    int numBlks = (m * n + MAX_BLK_THREADS - 1) / MAX_BLK_THREADS;
    dim3 gridElementWise(numBlks , 1 , 1);
    dim3 blkElementWise(MAX_BLK_THREADS , 1 , 1);
    addMatrixElementWise<<<gridElementWise,blkElementWise>>>(d_matA , d_matB , d_matC , m , n);
    cudaMemcpy(matC , d_matC , matSize , cudaMemcpyDeviceToHost);
    cudaFree(d_matC);
    
    // Row Wise Kernal Setup
    dim3 gridRowWise(1 , m , 1);
    dim3 blkRowWise(n , 1 , 1);
    addMatrixRowWise<<<gridRowWise,blkRowWise>>>(d_matA , d_matB , d_matD);
    cudaMemcpy(matD , d_matD , matSize , cudaMemcpyDeviceToHost);
    cudaFree(d_matD);

    // Col Wise Kernal Setup
    dim3 gridColWise(n, 1, 1);   
    dim3 blkColWise(m, 1, 1); 

    addMatrixColWise<<<gridColWise, blkColWise>>>(d_matA, d_matB, d_matE, m, n);
    cudaMemcpy(matE , d_matE , matSize , cudaMemcpyDeviceToHost);
    cudaFree(d_matE);

    // Device Memory Free
    cudaFree(d_matA);
    cudaFree(d_matB);

    // Display Result
    printf("\nMatrix C (Element Wise) :");
    displayMatrix(matC , m , n);

    printf("\nMatrix D (Row Wise) :");
    displayMatrix(matD , m , n);

    printf("\nMatrix E (Col Wise) :");
    displayMatrix(matE , m , n);

    // Host Memory Free
    free(matA);
    free(matB);
    free(matC);
    free(matD);
    free(matE);

    return 0;
}