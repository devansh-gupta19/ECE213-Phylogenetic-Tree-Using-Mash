#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "masterHeader.cuh"

// ============================================================================
// 1. SETUP KERNEL (Sequential - 1 Thread)
// ============================================================================
__global__ void InitNJMatrixKernel(int numSeqs, int numPairs, const int* d_pairA_idx, const int* d_pairB_idx, const float* d_out_D, float* d_distMatrix, bool* d_active) {
    int totalNodes = 2 * numSeqs - 1;

    for (int i = 0; i < numPairs; ++i) {
        int r = d_pairA_idx[i];
        int c = d_pairB_idx[i];
        float dist = d_out_D[i];
        d_distMatrix[r * totalNodes + c] = dist;
        d_distMatrix[c * totalNodes + r] = dist; 
    }
    for (int i = 0; i < totalNodes; ++i) {
        d_active[i] = (i < numSeqs);
    }
}

// ============================================================================
// 2. COMPUTE R KERNEL (Sequential - 1 Thread)
// ============================================================================
__global__ void ComputeNetDivergence_Kernel(float* d_distMatrix, bool* d_active, float* d_r, int totalNodes, int nextNodeId) {
    for (int i = 0; i < nextNodeId; ++i) {
        if (d_active[i]) {
            float r_val = 0.0f;
            for (int j = 0; j < nextNodeId; ++j) {
                if (d_active[j] && i != j) {
                    r_val += d_distMatrix[i * totalNodes + j];
                }
            }
            d_r[i] = r_val;
        }
    }
}

// ============================================================================
// 3. FIND MINIMUM Q & FINALIZE KERNEL (Sequential - 1 Thread)
// Replaces both Phase 1 and Phase 2 from the parallel version
// ============================================================================
__global__ void FindMinQ_And_Finalize_Kernel(float* d_distMatrix, bool* d_active, float* d_r, int* d_left_child, int* d_right_child, float* d_dist_left, float* d_dist_right, int totalNodes, int nextNodeId, int currentNodesCount, int* d_globalMinI, int* d_globalMinJ) {
    
    float minQ = FLT_MAX;
    int min_i = -1, min_j = -1;

    // Sequential search for minimum Q
    for (int i = 0; i < nextNodeId; ++i) {
        if (d_active[i]) {
            for (int j = i + 1; j < nextNodeId; ++j) {
                if (d_active[j]) {
                    float dist_ij = d_distMatrix[i * totalNodes + j];
                    float q = (currentNodesCount - 2) * dist_ij - d_r[i] - d_r[j];

                    if (q < minQ - 1e-6f) {
                        minQ = q; min_i = i; min_j = j;
                    } else if (fabsf(q - minQ) <= 1e-6f) {
                        if (i < min_i || (i == min_i && j < min_j)) {
                            minQ = q; min_i = i; min_j = j;
                        }
                    }
                }
            }
        }
    }

    // Finalize Topology
    d_globalMinI[0] = min_i;
    d_globalMinJ[0] = min_j;

    float dist_ij = d_distMatrix[min_i * totalNodes + min_j];
    float dist_i_u = 0.5f * dist_ij + (d_r[min_i] - d_r[min_j]) / (2.0f * (currentNodesCount - 2));
    float dist_j_u = dist_ij - dist_i_u;

    d_left_child[nextNodeId] = min_i; 
    d_right_child[nextNodeId] = min_j;
    d_dist_left[nextNodeId] = dist_i_u; 
    d_dist_right[nextNodeId] = dist_j_u;

    d_active[min_i] = false; 
    d_active[min_j] = false; 
    d_active[nextNodeId] = true;
}

// ============================================================================
// 4. UPDATE DISTANCE MATRIX KERNEL (Sequential - 1 Thread)
// ============================================================================
__global__ void UpdateDistanceMatrix_Kernel(float* d_distMatrix, bool* d_active, int totalNodes, int nextNodeId, int* d_globalMinI, int* d_globalMinJ) {
    int min_i = d_globalMinI[0];
    int min_j = d_globalMinJ[0];

    for (int k = 0; k < nextNodeId; ++k) {
        if (d_active[k] && k != min_i && k != min_j && k != nextNodeId) {
            float dist_ij = d_distMatrix[min_i * totalNodes + min_j];
            float dist_ik = d_distMatrix[min_i * totalNodes + k];
            float dist_jk = d_distMatrix[min_j * totalNodes + k];
            float dist_uk = 0.5f * (dist_ik + dist_jk - dist_ij);

            d_distMatrix[nextNodeId * totalNodes + k] = dist_uk;
            d_distMatrix[k * totalNodes + nextNodeId] = dist_uk;
        }
    }
}

// ============================================================================
// CALLER FUNCTION
// ============================================================================
void GpuAligner::NeighborJoiningCaller(int numSeqs, int numPairs, const int* d_pairA_idx, const int* d_pairB_idx, const float* d_out_D, int* h_left_child, int* h_right_child, float* h_dist_left, float* h_dist_right) {
    int totalNodes = 2 * numSeqs - 1;
    
    size_t matrixSize = totalNodes * totalNodes * sizeof(float);
    size_t floatArraySize = totalNodes * sizeof(float);
    size_t intArraySize = totalNodes * sizeof(int);
    size_t boolArraySize = totalNodes * sizeof(bool);

    float *d_distMatrix, *d_r, *d_dist_left, *d_dist_right;
    int *d_left_child, *d_right_child, *d_globalMinI, *d_globalMinJ;
    bool *d_active;

    cudaMalloc(&d_distMatrix, matrixSize); 
    cudaMalloc(&d_active, boolArraySize); 
    cudaMalloc(&d_r, floatArraySize);
    cudaMalloc(&d_left_child, intArraySize); 
    cudaMalloc(&d_right_child, intArraySize);
    cudaMalloc(&d_dist_left, floatArraySize); 
    cudaMalloc(&d_dist_right, floatArraySize);
    cudaMalloc(&d_globalMinI, sizeof(int));
    cudaMalloc(&d_globalMinJ, sizeof(int));
    
    cudaMemset(d_distMatrix, 0, matrixSize); 
    cudaMemset(d_left_child, -1, intArraySize); 
    cudaMemset(d_right_child, -1, intArraySize);
    cudaMemset(d_dist_left, 0, floatArraySize); 
    cudaMemset(d_dist_right, 0, floatArraySize);

    // 1. Initial Setup (Launched on exactly 1 block, 1 thread)
    InitNJMatrixKernel<<<1, 1>>>(numSeqs, numPairs, d_pairA_idx, d_pairB_idx, d_out_D, d_distMatrix, d_active);
    cudaDeviceSynchronize(); 

    int currentNodesCount = numSeqs;
    int nextNodeId = numSeqs;

    // 2. The CPU Orchestration Loop
    while (currentNodesCount > 2) {
        
        // Launch all kernels sequentially on 1 block, 1 thread
        ComputeNetDivergence_Kernel<<<1, 1>>>(d_distMatrix, d_active, d_r, totalNodes, nextNodeId);
        
        FindMinQ_And_Finalize_Kernel<<<1, 1>>>(d_distMatrix, d_active, d_r, d_left_child, d_right_child, d_dist_left, d_dist_right, totalNodes, nextNodeId, currentNodesCount, d_globalMinI, d_globalMinJ);
        
        UpdateDistanceMatrix_Kernel<<<1, 1>>>(d_distMatrix, d_active, totalNodes, nextNodeId, d_globalMinI, d_globalMinJ);

        // Synchronize to ensure the GPU finishes the iteration before the CPU loops
        cudaDeviceSynchronize();

        currentNodesCount--;
        nextNodeId++;
    }

    // 3. Copy Topology Back to Host
    cudaMemcpy(h_left_child, d_left_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right_child, d_right_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_left, d_dist_left, floatArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_right, d_dist_right, floatArraySize, cudaMemcpyDeviceToHost);

    // 4. Cleanup Memory
    cudaFree(d_distMatrix); cudaFree(d_active); cudaFree(d_r);
    cudaFree(d_left_child); cudaFree(d_right_child); cudaFree(d_dist_left); cudaFree(d_dist_right);
    cudaFree(d_globalMinI); cudaFree(d_globalMinJ);
}
