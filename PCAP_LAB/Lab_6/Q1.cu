#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void conv1D(double *N , double *M , double *P , int maskWidth , int width){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    double pVal = 0;
    int startPoint = i - (maskWidth / 2);

    for(int j = 0; j < maskWidth; j++){
        if(startPoint + j >=0 && startPoint + j < width){
            pVal += N[startPoint + j] * M[j];
        }
    }
    P[i] = pVal;
}

const int NUM_THREADS_PER_BLK = 64;

int main(void){
    // Allocate Host memory
    int n , m;
    printf("Enter N : ");
    scanf("%d" , &n);
    printf("Enter M : ");
    scanf("%d" , &m);
    double* vector1 = (double*) malloc(n * sizeof(double));
    double* vector2 = (double*) malloc(m * sizeof(double));
    double* vector3 = (double*) malloc(n * sizeof(double));
    for(int i = 0; i < n; i++){
        vector1[i] = i;
        vector3[i] = i;
    }
    for(int i = 0; i < m; i++){
        vector2[i] = i;
    }

    //Allocate Device Memory
    double * d_vector1 , *d_vector2 , *d_vector3;
    cudaMalloc((void**) &d_vector1 , n * sizeof(double));
    cudaMalloc((void**) &d_vector2 , m * sizeof(double));
    cudaMalloc((void**) &d_vector3 , n * sizeof(double));

    // Copy inputs to device
    cudaMemcpy(d_vector1 , vector1 , n * sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2 , vector2 , m * sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector3 , vector3 , n * sizeof(double) , cudaMemcpyHostToDevice);

    // Block Creation
    int numBlks = (n + NUM_THREADS_PER_BLK - 1) / NUM_THREADS_PER_BLK;

    //Launch add()kernal on gpu
    conv1D<<<numBlks , NUM_THREADS_PER_BLK>>>(d_vector1 , d_vector2 , d_vector3 , m , n);
    
    //Copy result back to host
    cudaMemcpy(vector3 , d_vector3 , n * sizeof(double) , cudaMemcpyDeviceToHost);
    
    //Print result
    printf("Conv1D Vector : %d blocks of %d threads\n" , numBlks , NUM_THREADS_PER_BLK);
    for(int i = 0; i < n;i++){
        printf("%lf " , vector3[i]);
    }
    printf("\n");

    //Cleanup
    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_vector3);
    free(vector1);
    free(vector2);
    free(vector3);
}