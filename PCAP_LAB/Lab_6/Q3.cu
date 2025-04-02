#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void oddEvenSort(int *a, int n, int *isSorted) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform multiple passes until isSorted is 1 (no swaps occur)
    while (*isSorted == 0) {
        // Reset isSorted to 1 at the beginning of each pass
        if (i == 0) {
            *isSorted = 1; // Assume sorted
        }
        __syncthreads();    // Ensure all threads have reset isSorted

        // Even phase
        if (i % 2 == 0 && i < n - 1) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
                atomicExch(isSorted, 0); // Set isSorted to 0
            }
        }
        __syncthreads(); // All threads must finish before moving on

        // Odd phase
        if (i % 2 == 1 && i < n - 1) {
            if (a[i] > a[i + 1]) {
                int temp = a[i];
                a[i] = a[i + 1];
                a[i + 1] = temp;
                atomicExch(isSorted, 0); // Set isSorted to 0
            }
        }
        __syncthreads(); // All threads must finish before moving on
    }
}

int main(int argc, char *argv[]) {
    int N;
    printf("Enter N : ");
    scanf("%d", &N);

    int *a, *d_a, *d_isSorted; // Host and device arrays + isSorted flag

    // Allocate arrays
    a = (int *)malloc(N * sizeof(int));
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_isSorted, sizeof(int)); // Allocate memory for isSorted flag on the device

    // Initialize array 'a' with some data (example)
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 1000; // Example random initialization, wider range
    }

    printf("Original array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    // Transfer data from host to device
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    // Set isSorted to 0 initially
    int isSorted_h = 0;
    cudaMemcpy(d_isSorted, &isSorted_h, sizeof(int), cudaMemcpyHostToDevice);

    oddEvenSort<<<1, N>>>(d_a, N, d_isSorted); // Single block, N threads

    // Transfer sorted array from device to host
    cudaMemcpy(a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print sorted array (optional)
    printf("Sorted array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");

    // Free memory
    free(a);
    cudaFree(d_a);
    cudaFree(d_isSorted);

    return 0;
}
