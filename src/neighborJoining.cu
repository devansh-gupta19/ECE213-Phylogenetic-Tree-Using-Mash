#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "masterHeader.cuh"

// Setup Kernel: Intialized distance matrix with MASH distances
__global__ void InitNJMatrixKernel(int numSeqs, int numPairs, const int* d_pairA_idx, const int* d_pairB_idx, const float* d_out_D, float* d_distMatrix, bool* d_active) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalNodes = 2 * numSeqs - 1;

    if (tid < numPairs) {
        int r = d_pairA_idx[tid];
        int c = d_pairB_idx[tid];
        float dist = d_out_D[tid];
        d_distMatrix[r * totalNodes + c] = dist;
        d_distMatrix[c * totalNodes + r] = dist; 
    }
    if (tid < totalNodes) {
        d_active[tid] = (tid < numSeqs);
    }
}

// Compute R Kernnel (Grid-Level Parallel)
__global__ void ComputeNetDivergence_Kernel(float* d_distMatrix, bool* d_active, float* d_r, int totalNodes, int nextNodeId) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < nextNodeId && d_active[i]) {
        float r_val = 0.0f;
        for (int j = 0; j < nextNodeId; ++j) {
            if (d_active[j] && i != j) {
                r_val += d_distMatrix[i * totalNodes + j];
            }
        }
        d_r[i] = r_val;
    }
}

// Find cell with minimum Q in one block of the matrix (Grid-Level Reduction)
__global__ void FindMinQ_Phase1_Kernel(float* d_distMatrix, bool* d_active, float* d_r, int totalNodes, int nextNodeId, int currentNodesCount, float* d_blockMinQ, int* d_blockMinI, int* d_blockMinJ) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_minQ = FLT_MAX;
    int local_min_i = -1, local_min_j = -1;

    // Each thread searches its assigned row
    if (i < nextNodeId && d_active[i]) {
        for (int j = i + 1; j < nextNodeId; ++j) {
            if (d_active[j]) {
                float dist_ij = d_distMatrix[i * totalNodes + j];
                float q = (currentNodesCount - 2) * dist_ij - d_r[i] - d_r[j];

                if (q < local_minQ - 1e-6f) {
                    local_minQ = q; local_min_i = i; local_min_j = j;
                } else if (fabsf(q - local_minQ) <= 1e-6f) {
                    if (i < local_min_i || (i == local_min_i && j < local_min_j)) {
                        local_minQ = q; local_min_i = i; local_min_j = j;
                    }
                }
            }
        }
    }

    // Block-level Shared Memory Reduction
    __shared__ float minQ_vals[256];
    __shared__ int minQ_i[256];
    __shared__ int minQ_j[256];

    minQ_vals[tid] = local_minQ;
    minQ_i[tid] = local_min_i;
    minQ_j[tid] = local_min_j;
    __syncthreads();

    // Parallel Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float q_stride = minQ_vals[tid + stride];
            float q_curr = minQ_vals[tid];
            if (q_stride < q_curr - 1e-6f) {
                minQ_vals[tid] = q_stride; minQ_i[tid] = minQ_i[tid + stride]; minQ_j[tid] = minQ_j[tid + stride];
            } else if (fabsf(q_stride - q_curr) <= 1e-6f) {
                int i_stride = minQ_i[tid + stride], j_stride = minQ_j[tid + stride];
                int i_curr = minQ_i[tid], j_curr = minQ_j[tid];
                if (i_stride < i_curr || (i_stride == i_curr && j_stride < j_curr)) {
                    minQ_vals[tid] = q_stride; minQ_i[tid] = i_stride; minQ_j[tid] = j_stride;
                }
            }
        }
        __syncthreads();
    }

    // Thread 0 of every block writes its winner to a global intermediate array
    if (tid == 0) {
        d_blockMinQ[blockIdx.x] = minQ_vals[0];
        d_blockMinI[blockIdx.x] = minQ_i[0];
        d_blockMinJ[blockIdx.x] = minQ_j[0];
    }
}

// Find the cell with minimum Q in the entire distance matrix (Finalize & Update Topology)
__global__ void FinalizeTopology_Kernel(float* d_distMatrix, bool* d_active, float* d_r, int* d_left_child, int* d_right_child, float* d_dist_left, float* d_dist_right, int totalNodes, int nextNodeId, int currentNodesCount, int numBlocksLaunched, float* d_blockMinQ, int* d_blockMinI, int* d_blockMinJ, int* d_globalMinI, int* d_globalMinJ) {
    int tid = threadIdx.x;
    
    __shared__ float minQ_vals[256]; 
    __shared__ int minQ_i[256];
    __shared__ int minQ_j[256];

    // Load the winners from Phase 1
    if (tid < numBlocksLaunched) {
        minQ_vals[tid] = d_blockMinQ[tid]; minQ_i[tid] = d_blockMinI[tid]; minQ_j[tid] = d_blockMinJ[tid];
    } else {
        minQ_vals[tid] = FLT_MAX; minQ_i[tid] = -1; minQ_j[tid] = -1;
    }
    __syncthreads();

    // Final Reduction to find the absolute minimum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float q_stride = minQ_vals[tid + stride];
            float q_curr = minQ_vals[tid];
            if (q_stride < q_curr - 1e-6f) {
                minQ_vals[tid] = q_stride; minQ_i[tid] = minQ_i[tid + stride]; minQ_j[tid] = minQ_j[tid + stride];
            } else if (fabsf(q_stride - q_curr) <= 1e-6f) {
                int i_stride = minQ_i[tid + stride], j_stride = minQ_j[tid + stride];
                int i_curr = minQ_i[tid], j_curr = minQ_j[tid];
                if (i_stride < i_curr || (i_stride == i_curr && j_stride < j_curr)) {
                    minQ_vals[tid] = q_stride; minQ_i[tid] = i_stride; minQ_j[tid] = j_stride;
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int min_i = minQ_i[0];
        int min_j = minQ_j[0];
        d_globalMinI[0] = min_i;
        d_globalMinJ[0] = min_j;

        float dist_ij = d_distMatrix[min_i * totalNodes + min_j];
        float dist_i_u = 0.5f * dist_ij + (d_r[min_i] - d_r[min_j]) / (2.0f * (currentNodesCount - 2));
        float dist_j_u = dist_ij - dist_i_u;

        d_left_child[nextNodeId] = min_i; d_right_child[nextNodeId] = min_j;
        d_dist_left[nextNodeId] = dist_i_u; d_dist_right[nextNodeId] = dist_j_u;

        d_active[min_i] = false; d_active[min_j] = false; d_active[nextNodeId] = true;
    }
}

// Update distance matrix(Grid-Level Parallel)
__global__ void UpdateDistanceMatrix_Kernel(float* d_distMatrix, bool* d_active, int totalNodes, int nextNodeId, int* d_globalMinI, int* d_globalMinJ) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    int min_i = d_globalMinI[0];
    int min_j = d_globalMinJ[0];

    if (k < nextNodeId && d_active[k] && k != min_i && k != min_j && k != nextNodeId) {
        float dist_ij = d_distMatrix[min_i * totalNodes + min_j];
        float dist_ik = d_distMatrix[min_i * totalNodes + k];
        float dist_jk = d_distMatrix[min_j * totalNodes + k];
        float dist_uk = 0.5f * (dist_ik + dist_jk - dist_ij);

        d_distMatrix[nextNodeId * totalNodes + k] = dist_uk;
        d_distMatrix[k * totalNodes + nextNodeId] = dist_uk;
    }
}


// Caller Function
void GpuAligner::NeighborJoiningCaller(int numSeqs, int numPairs, const int* d_pairA_idx, const int* d_pairB_idx, const float* d_out_D, int* h_left_child, int* h_right_child, float* h_dist_left, float* h_dist_right) {
    int totalNodes = 2 * numSeqs - 1;
    
    size_t matrixSize = totalNodes * totalNodes * sizeof(float);
    size_t floatArraySize = totalNodes * sizeof(float);
    size_t intArraySize = totalNodes * sizeof(int);
    size_t boolArraySize = totalNodes * sizeof(bool);

    float *d_distMatrix, *d_r, *d_dist_left, *d_dist_right;
    int *d_left_child, *d_right_child;
    bool *d_active;

    cudaMalloc(&d_distMatrix, matrixSize); cudaMalloc(&d_active, boolArraySize); cudaMalloc(&d_r, floatArraySize);
    cudaMalloc(&d_left_child, intArraySize); cudaMalloc(&d_right_child, intArraySize);
    cudaMalloc(&d_dist_left, floatArraySize); cudaMalloc(&d_dist_right, floatArraySize);
    
    cudaMemset(d_distMatrix, 0, matrixSize); cudaMemset(d_left_child, -1, intArraySize); cudaMemset(d_right_child, -1, intArraySize);
    cudaMemset(d_dist_left, 0, floatArraySize); cudaMemset(d_dist_right, 0, floatArraySize);

    // Initial Setup
    int maxThreadsNeeded = (numPairs > totalNodes) ? numPairs : totalNodes;
    int setupBlockSize = 256;
    InitNJMatrixKernel<<<(maxThreadsNeeded + setupBlockSize - 1) / setupBlockSize, setupBlockSize>>>(numSeqs, numPairs, d_pairA_idx, d_pairB_idx, d_out_D, d_distMatrix, d_active);
    cudaDeviceSynchronize(); 

    // Allocate intermediate reduction arrays (sized to hold 1 winner per block)
    int maxBlocks = (totalNodes + 255) / 256;
    float* d_blockMinQ; int *d_blockMinI, *d_blockMinJ, *d_globalMinI, *d_globalMinJ;
    cudaMalloc(&d_blockMinQ, maxBlocks * sizeof(float));
    cudaMalloc(&d_blockMinI, maxBlocks * sizeof(int));
    cudaMalloc(&d_blockMinJ, maxBlocks * sizeof(int));
    cudaMalloc(&d_globalMinI, sizeof(int));
    cudaMalloc(&d_globalMinJ, sizeof(int));

    int currentNodesCount = numSeqs;
    int nextNodeId = numSeqs;

    // CPU calls the kernels until all nodes have been assigned to the tree topology
    while (currentNodesCount > 2) {
        int threadsPerBlock = 256;
        int numBlocks = (nextNodeId + threadsPerBlock - 1) / threadsPerBlock;

        // Fire all 4 kernels asynchronously into the CUDA Stream
        ComputeNetDivergence_Kernel<<<numBlocks, threadsPerBlock>>>(d_distMatrix, d_active, d_r, totalNodes, nextNodeId);
        
        FindMinQ_Phase1_Kernel<<<numBlocks, threadsPerBlock>>>(d_distMatrix, d_active, d_r, totalNodes, nextNodeId, currentNodesCount, d_blockMinQ, d_blockMinI, d_blockMinJ);
        
        FinalizeTopology_Kernel<<<1, 256>>>(d_distMatrix, d_active, d_r, d_left_child, d_right_child, d_dist_left, d_dist_right, totalNodes, nextNodeId, currentNodesCount, numBlocks, d_blockMinQ, d_blockMinI, d_blockMinJ, d_globalMinI, d_globalMinJ);
        
        UpdateDistanceMatrix_Kernel<<<numBlocks, threadsPerBlock>>>(d_distMatrix, d_active, totalNodes, nextNodeId, d_globalMinI, d_globalMinJ);

        // Synchronize to ensure the GPU finishes the iteration before the CPU loops
        cudaDeviceSynchronize();

        currentNodesCount--;
        nextNodeId++;
    }

    // Copy Topology Back to Host
    cudaMemcpy(h_left_child, d_left_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right_child, d_right_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_left, d_dist_left, floatArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_right, d_dist_right, floatArraySize, cudaMemcpyDeviceToHost);

    // Cleanup Memory
    cudaFree(d_distMatrix); cudaFree(d_active); cudaFree(d_r);
    cudaFree(d_left_child); cudaFree(d_right_child); cudaFree(d_dist_left); cudaFree(d_dist_right);
    cudaFree(d_blockMinQ); cudaFree(d_blockMinI); cudaFree(d_blockMinJ); cudaFree(d_globalMinI); cudaFree(d_globalMinJ);
}