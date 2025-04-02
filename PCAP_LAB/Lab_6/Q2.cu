#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// Kernel to merge two sorted subarrays
__global__ void merge(int *arr, int *temp, int n, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate left, mid, and right indices for this thread
    int left_start = idx * size * 2;
    if (left_start >= n) return;

    int mid = min(left_start + size - 1, n - 1);
    int right_end = min(left_start + 2 * size - 1, n - 1);

    // Merge two sorted subarrays
    int i = left_start;
    int j = mid + 1;
    int k = left_start;

    while (i <= mid && j <= right_end) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    while (i <= mid) {
        temp[k++] = arr[i++];
    }
    while (j <= right_end) {
        temp[k++] = arr[j++];
    }

    // Copy merged data back to the original array
    for (i = left_start; i <= right_end; i++) {
        arr[i] = temp[i];
    }
}

// Kernel to perform insertion sort on small segments
__global__ void insertionSort(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * BLOCK_SIZE;

    if (start >= n) return;

    // Perform insertion sort on this segment
    for (int i = start + 1; i < min(start + BLOCK_SIZE, n); i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= start && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Host function to perform iterative merge sort
void iterativeMergeSort(int *d_arr, int *d_temp, int n) {
    // Step 1: Sort small segments using insertion sort
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    insertionSort<<<numBlocks, BLOCK_SIZE>>>(d_arr, n);
    cudaDeviceSynchronize(); // Synchronize after sorting small segments

    // Step 2: Iteratively merge sorted segments
    for (int size = BLOCK_SIZE; size < n; size *= 2) {
        numBlocks = (n + size * 2 - 1) / (size * 2);
        merge<<<numBlocks, BLOCK_SIZE>>>(d_arr, d_temp, n, size);
        cudaDeviceSynchronize(); // Synchronize after each merge step
    }
}

int main() {
    int N;
    printf("Enter N : ");
    scanf("%d" , &N);
    int *h_arr = (int *)malloc(N * sizeof(int)); // Host array
    int *d_arr, *d_temp;                         // Device arrays

    // Initialize array with random values
    printf("Original Array : \n");
    for (int i = 0; i < N; i++) {
        h_arr[i] = rand() % 1000;
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Allocate memory on the device
    cudaMalloc((void **)&d_arr, N * sizeof(int));
    cudaMalloc((void **)&d_temp, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Perform iterative merge sort on GPU
    iterativeMergeSort(d_arr, d_temp, N);

    // Copy data from device back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted Array : \n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_arr);
    cudaFree(d_temp);
    free(h_arr);

    printf("Sorting complete.\n");
    return 0;
}
