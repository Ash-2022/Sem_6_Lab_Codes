#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void add(int*a , int*b , int*c){
    *c = *a + *b;
}

int main(void){
    int a , b , c;
    int *d_a , *d_b , *d_c;
    int size = sizeof(int);

    // Allocate Space for device copies
    cudaMalloc((void**) &d_a , size);
    cudaMalloc((void**) &d_b , size);
    cudaMalloc((void**) &d_c , size);
    //Setup Vals
    a = 3;
    b= 5;
    // Copy inputs to device
    cudaMemcpy(d_a , &a , size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_b , &b , size , cudaMemcpyHostToDevice);
    cudaMemcpy(d_c , &c , size , cudaMemcpyHostToDevice);
    // Launch add() kernal on gpu
    add<<<1,1>>>(d_a , d_b , d_c);
    //COpy result back to host
    cudaMemcpy(&c , d_c , size , cudaMemcpyDeviceToHost);
    printf("Result : %d\n" , c);
    //Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}