#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

__device__ int fn(int* arg) {
    return 99;
}

__device__ void other_fonction(int* arg) {
    *arg = fn(arg);
}

__global__ void fonction(int* arg) {
    other_fonction(arg);
}

int main() {
    int* a;
    cudaMallocManaged(&a, sizeof(int));
    *a = 3;
    fonction<<<2, 512>>>(a);
    cudaDeviceSynchronize();
    printf("%d", *a);
    return 0;
}