#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void calcSine(double*a , double*c , int size){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < size) c[id] = sin(a[id]);
}

const int NUM_THREADS_PER_BLK = 64;
const double PI = 3.1414;

int main(void){
    // Allocate Host memory
    int size;
    printf("Enter Num of elements of vector : ");
    scanf("%d" , &size);
    double* vector1 = (double*) malloc(size * sizeof(double));
    double* vector2 = (double*) malloc(size * sizeof(double));
    for(int i = 0; i < size; i++){
        vector1[i] = (i/(double)180) * PI;
    }
    int vectorSize = size * sizeof(double);

    //Allocate Device Memory
    double * d_vector1 , *d_vector2;
    cudaMalloc((void**) &d_vector1 , vectorSize);
    cudaMalloc((void**) &d_vector2 , vectorSize);

    // Copy inputs to device
    cudaMemcpy(d_vector1 , vector1 , vectorSize , cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2 , vector2 , vectorSize , cudaMemcpyHostToDevice);

    // Block Creation
    int numBlks = size / NUM_THREADS_PER_BLK + 1;

    //Launch add()kernal on gpu
    calcSine<<<numBlks , NUM_THREADS_PER_BLK>>>(d_vector1 , d_vector2 , size);
    
    //Copy result back to host
    cudaMemcpy(vector2 , d_vector2 , vectorSize , cudaMemcpyDeviceToHost);
    
    //Print result
    printf("Sine Vector : %d blocks of 256 threads\n" , numBlks);
    for(int i = 0; i < size;i++){
        printf("%lf " , vector2[i]);
    }
    printf("\n");

    //Cleanup
    cudaFree(d_vector1);
    cudaFree(d_vector2);
    free(vector1);
    free(vector2);
}