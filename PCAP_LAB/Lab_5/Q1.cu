#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void addVector(int*a , int*b , int*c , int size){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < size) c[id] = a[id] + b[id];
}

int main(void){
    // Allocate Host memory
    int size;
    printf("Enter Num of elements of vector : ");
    scanf("%d" , &size);
    int* vector1 = (int*) malloc(size * sizeof(int));
    int* vector2 = (int*) malloc(size * sizeof(int));
    int* vector3 = (int*) malloc(size * sizeof(int));
    int* vector4 = (int*) malloc(size * sizeof(int));
    if(size < 5){
        printf("Vector 1 : \n");
        for(int i = 0; i < size; i++){
            printf("Enter ele %d : " , (i+1));
            scanf("%d" , &vector1[i]);
        }
        printf("Vector 2 : \n");
        for(int i = 0; i < size; i++){
            printf("Enter ele %d : " , (i+1));
            scanf("%d" , &vector2[i]);
        }
    }
    else{
        for(int i = 0; i < size; i++){
            vector1[i] = i;
            vector2[i] = size + i;
        }
    }
    int vectorSize = size * sizeof(int);

    //Allocate Device Memory
    int * d_vector1 , *d_vector2 , *d_vector3 , *d_vector4;
    cudaMalloc((void**) &d_vector1 , vectorSize);
    cudaMalloc((void**) &d_vector2 , vectorSize);
    cudaMalloc((void**) &d_vector3 , vectorSize);
    cudaMalloc((void**) &d_vector4 , vectorSize);

    // Copy inputs to device
    cudaMemcpy(d_vector1 , vector1 , vectorSize , cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2 , vector2 , vectorSize , cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector3 , vector3 , vectorSize , cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector4 , vector4 , vectorSize , cudaMemcpyHostToDevice);

    // Block Creation
    int blkSize = 0;
    if(!(size % 32)) blkSize = size;
    else{
        int a = size % 32;
        blkSize = size + 32 - a;
    }

    //Launch add()kernal on gpu
    addVector<<<blkSize , 1>>>(d_vector1 , d_vector2 , d_vector3 , size);
    addVector<<<1,blkSize>>>(d_vector1 , d_vector2 , d_vector4 , size);
    
    //Copy result back to host
    cudaMemcpy(vector3 , d_vector3 , vectorSize , cudaMemcpyDeviceToHost);
    cudaMemcpy(vector4 , d_vector4 , vectorSize , cudaMemcpyDeviceToHost);
    
    //Print result
    printf("Vector 3 : N Threads\n");
    for(int i = 0; i < size;i++){
        printf("%d " , vector3[i]);
    }
    printf("\n");
    printf("Vector 4 : Blk size = N\n");
    for(int i = 0; i < size;i++){
        printf("%d " , vector4[i]);
    }
    printf("\n");

    //Cleanup
    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_vector3);
    cudaFree(d_vector4);
    free(vector1);
    free(vector2);
    free(vector3);
    free(vector4);
}