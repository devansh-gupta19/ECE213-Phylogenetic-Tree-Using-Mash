#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "masterHeader.h"

// ============================================================================
// Kernel 1: Parallel Setup - Maps 1D Mash output to 2D symmetric matrix
// ============================================================================
__global__ void InitNJMatrixKernel(
    int numSeqs,
    int numPairs,
    const int* d_pairA_idx,
    const int* d_pairB_idx,
    const float* d_out_D,
    float* d_distMatrix,
    bool* d_active) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalNodes = 2 * numSeqs - 1;

    // 1. Populate the 2D symmetric distance matrix from the 1D arrays
    if (tid < numPairs) {
        int r = d_pairA_idx[tid];
        int c = d_pairB_idx[tid];
        float dist = d_out_D[tid];
        
        d_distMatrix[r * totalNodes + c] = dist;
        d_distMatrix[c * totalNodes + r] = dist; // Symmetric
    }

    // 2. Initialize the active mask (only original sequences 0 to numSeqs-1 are active initially)
    if (tid < totalNodes) {
        d_active[tid] = (tid < numSeqs);
    }
}

// ============================================================================
// Kernel 2: Sequential Neighbor Joining Algorithm
// ============================================================================
__global__ void SequentialNeighborJoiningKernel(
    float* d_distMatrix,  
    bool* d_active,       
    float* d_r,           
    int* d_left_child,    
    int* d_right_child,   
    float* d_dist_left,   
    float* d_dist_right,  
    int numSeqs)
{
    // Restrict execution to a single thread
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int totalNodes = 2 * numSeqs - 1;
    int currentNodesCount = numSeqs;
    int nextNodeId = numSeqs;

    while (currentNodesCount > 2) {
        // 1. Calculate Net Divergence r[i]
        for (int i = 0; i < nextNodeId; ++i) {
            d_r[i] = 0.0f;
            if (!d_active[i]) continue;
            
            for (int j = 0; j < nextNodeId; ++j) {
                if (d_active[j] && i != j) {
                    d_r[i] += d_distMatrix[i * totalNodes + j];
                }
            }
        }

        // 2. Calculate Q-matrix and find the minimum pair
        float minQ = FLT_MAX;
        int min_i = -1, min_j = -1;

        for (int i = 0; i < nextNodeId; ++i) {
            if (!d_active[i]) continue;
            for (int j = i + 1; j < nextNodeId; ++j) {
                if (!d_active[j]) continue;

                float dist_ij = d_distMatrix[i * totalNodes + j];
                float q = (currentNodesCount - 2) * dist_ij - d_r[i] - d_r[j];

                if (q < minQ) {
                    minQ = q;
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // 3. Compute Branch Lengths and Update Tree Topology
        float dist_ij = d_distMatrix[min_i * totalNodes + min_j];
        float dist_i_u = 0.5f * dist_ij + (d_r[min_i] - d_r[min_j]) / (2.0f * (currentNodesCount - 2));
        float dist_j_u = dist_ij - dist_i_u;

        d_left_child[nextNodeId] = min_i;
        d_right_child[nextNodeId] = min_j;
        d_dist_left[nextNodeId] = dist_i_u;
        d_dist_right[nextNodeId] = dist_j_u;

        // 4. Update the Distance Matrix for the new node
        for (int k = 0; k < nextNodeId; ++k) {
            if (d_active[k] && k != min_i && k != min_j) {
                float dist_ik = d_distMatrix[min_i * totalNodes + k];
                float dist_jk = d_distMatrix[min_j * totalNodes + k];
                float dist_uk = 0.5f * (dist_ik + dist_jk - dist_ij);

                d_distMatrix[nextNodeId * totalNodes + k] = dist_uk;
                d_distMatrix[k * totalNodes + nextNodeId] = dist_uk;
            }
        }

        // 5. Update Active Masks
        d_active[min_i] = false;
        d_active[min_j] = false;
        d_active[nextNodeId] = true;

        currentNodesCount--;
        nextNodeId++;
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