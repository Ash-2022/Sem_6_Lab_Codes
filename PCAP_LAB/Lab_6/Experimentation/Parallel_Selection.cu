#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define BLK_SIZE 1024

__host__ void populateVector(int **v , int l){
    for(int i = 0; i < l; i++){
        (*v)[i] = rand() % 1000;
    }
}

__global__ void sortVector(int *d_in , int *d_out , int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n){
        int finalPos = 0;
        int ele = d_in[idx];
        for(int i = 0; i < n; i++){
            if((ele < d_in[i]) || (i < idx && ele == d_in[i])) finalPos++;
        }
        d_out[finalPos] = ele;
    }
}

int main(){
    int n;
    printf("Enter Size of Array : ");
    scanf("%d" , &n);

    int size = n * sizeof(int);

    // Allocate CPU memory
    int *input = (int*) malloc(size);
    int *output = (int*) malloc(size);
    populateVector(&input , n);

    // Create CUDA Events
    cudaEvent_t start , stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start Timer
    printf("Started GPU\n");
    cudaEventRecord(start , 0);

    // Allocate GPU memory
    int *d_input , *d_output;
    cudaMalloc(&d_input , size);
    cudaMalloc(&d_output , size);

    // Transfer Data to GPU
    cudaMemcpy(d_input , input , size , cudaMemcpyHostToDevice);

    // Grid & Block Creation
    int numBlks = ceil((float)n / BLK_SIZE);
    dim3 dimGrid(numBlks , 1 , 1);
    dim3 dimBlk(BLK_SIZE , 1 , 1);

    // Kernal Function
    sortVector<<<dimGrid , dimBlk>>>(d_input , d_output , n);

    // Transfer Back to CPU
    cudaMemcpy(output , d_output , size , cudaMemcpyDeviceToHost);

    // Free GPU Memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Stop Timer
    cudaEventRecord(stop , 0);
    cudaEventSynchronize(stop);

    // Print Sorted Vector
    printf("Sorted Array \n");
    
    // Display elapsed Time in ms
    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime , start , stop);
    printf("GPU time is %.4f ms\n" , elapsedTime);

    // Free CPU Memory
    free(output);
    free(input);
    return 0;
}
