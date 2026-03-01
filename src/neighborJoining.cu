#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "masterHeader.h"

__global__ void SequentialInitNJMatrixKernel(
    int numSeqs,
    int numPairs,
    const int* d_pairA_idx,
    const int* d_pairB_idx,
    const float* d_out_D,
    float* d_distMatrix,
    bool* d_active) 
{
    // Restrict execution to exactly one thread
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int totalNodes = 2 * numSeqs - 1;

    // 1. Populate the 2D symmetric distance matrix using a standard loop
    for (int i = 0; i < numPairs; ++i) {
        int r = d_pairA_idx[i];
        int c = d_pairB_idx[i];
        float dist = d_out_D[i];
        
        d_distMatrix[r * totalNodes + c] = dist;
        d_distMatrix[c * totalNodes + r] = dist; // Symmetric
    }

    // 2. Initialize the active mask sequentially
    for (int i = 0; i < totalNodes; ++i) {
        d_active[i] = (i < numSeqs);
    }
}

void GpuAligner::NeighborJoiningCaller(
    int numSeqs, 
    int numPairs, 
    const int* d_pairA_idx,  // Input: DEVICE pointer
    const int* d_pairB_idx,  // Input: DEVICE pointer
    const float* d_out_D,    // Input: DEVICE pointer
    int* h_left_child,       // Output: HOST pointer
    int* h_right_child,      // Output: HOST pointer
    float* h_dist_left,      // Output: HOST pointer
    float* h_dist_right)     // Output: HOST pointer
{
    // A fully resolved tree with N leaves has 2N-1 total nodes
    int totalNodes = 2 * numSeqs - 1;
    
    // Memory sizes
    size_t matrixSize = totalNodes * totalNodes * sizeof(float);
    size_t floatArraySize = totalNodes * sizeof(float);
    size_t intArraySize = totalNodes * sizeof(int);
    size_t boolArraySize = totalNodes * sizeof(bool);

    // ========================================================================
    // 1. Allocate Device Memory for Tree Structures
    // ========================================================================
    float *d_distMatrix, *d_r, *d_dist_left, *d_dist_right;
    int *d_left_child, *d_right_child;
    bool *d_active;

    cudaMalloc(&d_distMatrix, matrixSize);
    cudaMalloc(&d_active, boolArraySize);
    cudaMalloc(&d_r, floatArraySize);
    cudaMalloc(&d_left_child, intArraySize);
    cudaMalloc(&d_right_child, intArraySize);
    cudaMalloc(&d_dist_left, floatArraySize);
    cudaMalloc(&d_dist_right, floatArraySize);

    // Initialize topology arrays to default values on device
    cudaMemset(d_distMatrix, 0, matrixSize);
    cudaMemset(d_left_child, -1, intArraySize);
    cudaMemset(d_right_child, -1, intArraySize);
    cudaMemset(d_dist_left, 0, floatArraySize);
    cudaMemset(d_dist_right, 0, floatArraySize);

    // ========================================================================
    // 2. Launch Setup Kernel (Parallel)
    // ========================================================================
    // Calculate thread count needed: max of numPairs or totalNodes
    int maxThreadsNeeded = (numPairs > totalNodes) ? numPairs : totalNodes;
    int blockSize = 256;
    int numBlocks = (maxThreadsNeeded + blockSize - 1) / blockSize;

    InitNJMatrixKernel<<<numBlocks, blockSize>>>(
        numSeqs, numPairs, d_pairA_idx, d_pairB_idx, d_out_D, d_distMatrix, d_active
    );
    cudaDeviceSynchronize(); // Ensure matrix is built before NJ starts

    // ========================================================================
    // 3. Launch the Sequential NJ Kernel (1 Block, 1 Thread)
    // ========================================================================
    SequentialNeighborJoiningKernel<<<1, 1>>>(
        d_distMatrix, d_active, d_r, 
        d_left_child, d_right_child, 
        d_dist_left, d_dist_right, 
        numSeqs
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Sequential NJ kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); // Wait for the single thread to finish the tree

    // ========================================================================
    // 4. Copy Tree Topology Back to Host
    // ========================================================================
    // Copy the small topology arrays directly into the provided host pointers
    cudaMemcpy(h_left_child, d_left_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right_child, d_right_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_left, d_dist_left, floatArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_right, d_dist_right, floatArraySize, cudaMemcpyDeviceToHost);

    // ========================================================================
    // 5. Cleanup
    // ========================================================================
    // Free device memory allocated inside this function
    cudaFree(d_distMatrix);
    cudaFree(d_active);
    cudaFree(d_r);
    cudaFree(d_left_child);
    cudaFree(d_right_child);
    cudaFree(d_dist_left);
    cudaFree(d_dist_right);
}