#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// Function to find minimum bits required to represent a number
__device__ int findMinBits(int num) {
    if (num == 0) return 1;
    
    int bits = 0;
    while (num > 0) {
        bits++;
        num >>= 1;
    }
    return bits;
}

// Function to convert an integer to its binary representation
__device__ void decimalToBinary(int decimal, char* binary, int minBits) {
    unsigned int mask = (1U << minBits) - 1;
    int complementValue = (~decimal) & mask;
    
    int startIdx = 0;
    bool foundMSB = false;
    
    for (int i = minBits - 1; i >= 0; i--) {
        char bit = ((complementValue >> i) & 1) + '0';
        if (bit == '1' || foundMSB) {
            binary[startIdx++] = bit;
            foundMSB = true;
        }
    }
    binary[startIdx] = '\0';
}

// Custom function to convert a string to an integer in device code
__device__ int strToInt(const char* str) {
    int result = 0;
    bool isNegative = false;
    int i = 0;

    if (str[0] == '-') {
        isNegative = true;
        i++;
    }

    for (; str[i] != '\0'; i++) {
        result = result * 10 + (str[i] - '0');
    }

    return isNegative ? -result : result;
}

// Kernel function to transform matrix (convert decimal to binary for non-border elements)
__global__ void transformMatrixKernel(char *d_outputMatrix, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1) {
        int index = i * cols + j;

        // Convert string to integer
        int value = strToInt(&d_outputMatrix[index * 33]);

        // Convert to binary
        int minBits = findMinBits(value);
        char binaryStr[33] = {0};
        decimalToBinary(value, binaryStr, minBits);

        // Copy binary string to output matrix
        for (int k = 0; k < 33; k++) {
            d_outputMatrix[index * 33 + k] = binaryStr[k];
        }
    }
}

// Host function to handle memory and data transfer
void transformMatrix(float *h_inputMatrix, char *h_outputMatrix, int rows, int cols) {
    char *d_outputMatrix;
    size_t outputSize = rows * cols * 33 * sizeof(char);

    // Prepare host output matrix with integer strings
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = (i * cols + j) * 33;
            sprintf(&h_outputMatrix[index], "%d", (int)h_inputMatrix[i * cols + j]);
        }
    }

    // Allocate device memory and copy prepared host output matrix to GPU
    cudaMalloc((void**)&d_outputMatrix, outputSize);
    cudaMemcpy(d_outputMatrix, h_outputMatrix, outputSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);

    // Launch kernel (only modifies non-border elements)
    transformMatrixKernel<<<gridSize, blockSize>>>(d_outputMatrix, rows, cols);
    cudaDeviceSynchronize();

    // Copy transformed matrix back to host
    cudaMemcpy(h_outputMatrix, d_outputMatrix, outputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_outputMatrix);
}

int main() {
    int rows = 4, cols = 4;
    float inputMatrix[] = {
        1, 2, 3, 4,
        6, 5, 8, 3,
        2, 4, 10, 1,
        9, 1, 2, 5
    };

    // Allocate output matrix with space for binary strings
    char outputMatrix[(rows * cols * 33) + 1];
    memset(outputMatrix, 0, sizeof(outputMatrix));

    printf("Original Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", (int)inputMatrix[i * cols + j]);
        }
        printf("\n");
    }

    transformMatrix(inputMatrix, outputMatrix, rows, cols);

    printf("\nTransformed Matrix (Binary Representation):\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%s ", &outputMatrix[(i * cols + j) * 33]);
        }
        printf("\n");
    }

    return 0;
}
