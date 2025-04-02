#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEN 100
/*
    PCAP
    PCAPPCAPCP
    i = 0 , starts at 0
    i = 1 , starts at 4
    i = 2 , starts at 7
    i = 3 , starts at 9
*/

__global__ void replicateLetters(char * d_i , char* d_o){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = blockDim.x;
    if(idx < blockDim.x){
        int startIdx = 0;
        for(int i = 0; i < idx; i++){
            startIdx += (temp);
            temp--;
        }
        for(int i = 0; i < (blockDim.x - idx); i++){
            d_o[startIdx + i] = d_i[i];
        }
    }
}

int main() {
    char *input = (char*) malloc(sizeof(char) * MAX_LEN);

    printf("Enter a string: ");
    fgets(input, MAX_LEN, stdin);
    int n = strlen(input);
    if (input[n - 1] == '\n') input[n - 1] = '\0';
    n--;
    int out_n = ((n*(n+1)) / 2);
    char * output = (char*) malloc(sizeof(char) * out_n);

    // Allocate device memory for tokens and word to search
    char *d_input , *d_output;

    cudaMalloc(&d_input, sizeof(char) * n);
    cudaMalloc(&d_output, sizeof(char) * out_n);

    // Copy the input word to device memory
    cudaMemcpy(d_input, input, sizeof(char) * n, cudaMemcpyHostToDevice);

    // Launch kernel to count occurrences
    replicateLetters<<<1, n>>>(d_input, d_output);

    // Copy the result back to host
    cudaMemcpy(output, d_output, out_n, cudaMemcpyDeviceToHost);

    printf("Output = %s\n", output);

    cudaFree(d_input);  // Free the array of pointers on device
    cudaFree(d_output);

    free(input);
    free(output);
    return 0;
}
