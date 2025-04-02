#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEN 100
#define MAX_TOKENS 100

// Tokenize a string into an array of tokens
char** strTokenize(char *str, int *numTokens) {
    char **strArray = (char**) malloc(sizeof(char*) * MAX_TOKENS);
    int idx = 0;
    char *token = strtok(str, " ");
    while (token != NULL) {
        strArray[idx] = (char*)malloc(sizeof(char) * (strlen(token) + 1));
        strcpy(strArray[idx], token);
        idx++;
        token = strtok(NULL, " ");
    }
    *numTokens = idx;
    return strArray;
}

// Device function to compare two strings
__device__ int stringCompare(const char* str1, const char* str2) {
    int i = 0;
    while (str1[i] != '\0' && str2[i] != '\0') {
        if (str1[i] != str2[i]) {
            return 0; // Return 0 for false
        }
        i++;
    }
    return (str1[i] == str2[i]) ? 1 : 0; // Return 1 for true, 0 for false
}

// Kernel to count occurrences of a word in the tokenized string
__global__ void countWords(char** buf, int numTokens, char* str, int *d_count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numTokens) {
        // Compare each token with the target string
        if (stringCompare(buf[idx], str)) {
            atomicAdd(d_count, 1);
        }
    }
}

int main() {
    char *input = (char*) malloc(sizeof(char) * MAX_LEN);
    char *inputWord = (char*) malloc(sizeof(char) * MAX_LEN);

    printf("Enter a string: ");
    fgets(input, MAX_LEN, stdin);
    printf("Enter a string to match : ");
    fgets(inputWord, MAX_LEN, stdin);

    int l1 = strlen(input);
    int l2 = strlen(inputWord);
    int count = 0;

    if (input[l1 - 1] == '\n') input[l1 - 1] = '\0';
    if (inputWord[l2 - 1] == '\n') inputWord[l2 - 1] = '\0';

    int numTokens = 0;
    char **tokens = strTokenize(input, &numTokens);

    // Allocate device memory for tokens and word to search
    char **d_tokens;
    char *d_str;
    int *d_count;

    cudaMalloc(&d_tokens, sizeof(char*) * numTokens);
    cudaMalloc(&d_str, sizeof(char) * (l2 + 1));
    cudaMalloc(&d_count, sizeof(int));

    // Allocate memory for each token on the device
    char** d_token_ptrs = (char**)malloc(numTokens * sizeof(char*)); // Create an array to hold pointers to device tokens
    for (int i = 0; i < numTokens; i++) {
        // Allocate memory for each token on the device
        char* d_token;
        cudaMalloc(&d_token, sizeof(char) * (strlen(tokens[i]) + 1));
        cudaMemcpy(d_token, tokens[i], sizeof(char) * (strlen(tokens[i]) + 1), cudaMemcpyHostToDevice);
        d_token_ptrs[i] = d_token; // Store the device pointer in the array
    }

    // Copy the input word to device memory
    cudaMemcpy(d_str, inputWord, sizeof(char) * (l2 + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    // Copy the pointers array to the device
    cudaMemcpy(d_tokens, d_token_ptrs, sizeof(char*) * numTokens, cudaMemcpyHostToDevice);

    // Launch kernel to count occurrences
    countWords<<<1, numTokens>>>(d_tokens, numTokens, d_str, d_count);

    // Copy the result back to host
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Count = %d\n", count);

    // Free allocated memory
    for (int i = 0; i < numTokens; i++) {
        cudaFree(d_token_ptrs[i]);  // Free each token's memory on the device
        free(tokens[i]);  // Free token memory on the host
    }
    free(tokens);
    free(input);
    free(inputWord);

    cudaFree(d_tokens);  // Free the array of pointers on device
    cudaFree(d_str);
    cudaFree(d_count);
    free(d_token_ptrs);  // Free the array holding device token pointers on host

    return 0;
}
