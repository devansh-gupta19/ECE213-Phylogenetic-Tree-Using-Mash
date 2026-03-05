#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include "masterHeader.cuh"

// ============================================================================
// Kernel 1: Parallel Setup
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

    // Map 1D pairwise array to 2D symmetric matrix in parallel
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

// ============================================================================
// Kernel 2: Hyper-Optimized Parallel Neighbor Joining 
// ============================================================================
__global__ void OptimizedNeighborJoiningKernel(
    float* d_distMatrix,  
    bool* d_active,       
    float* d_r,           
    int* d_left_child,    
    int* d_right_child,   
    float* d_dist_left,   
    float* d_dist_right,  
    int numSeqs)
{
    int tid = threadIdx.x;
    int totalNodes = 2 * numSeqs - 1;
    int currentNodesCount = numSeqs;
    int nextNodeId = numSeqs;

    // ------------------------------------------------------------------------
    // OPTIMIZATION 1: Static Shared Memory for Parallel Reduction
    // ------------------------------------------------------------------------
    __shared__ float minQ_vals[512];
    __shared__ int minQ_i[512];
    __shared__ int minQ_j[512];

    // ------------------------------------------------------------------------
    // OPTIMIZATION 2: Dynamic Shared Memory Caching for Arrays
    // ------------------------------------------------------------------------
    extern __shared__ float dynamic_smem[];
    float* s_r = dynamic_smem; 
    
    // OPTIMIZATION 3: Word-aligned memory. Cast bools to ints in Shared Mem
    int* s_active = (int*)&s_r[totalNodes]; 

    // Collaboratively load the global active mask into blazing-fast shared memory
    for(int i = tid; i < totalNodes; i += blockDim.x) {
        s_active[i] = (int)d_active[i];
    }
    __syncthreads();

    while (currentNodesCount > 2) {
        
        // ---------------------------------------------------------
        // Step 1: Calculate Net Divergence r[i] in Parallel
        // ---------------------------------------------------------
        for (int i = tid; i < nextNodeId; i += blockDim.x) {
            float r_val = 0.0f;
            if (s_active[i]) {
                for (int j = 0; j < nextNodeId; ++j) {
                    if (s_active[j] && i != j) {
                        r_val += d_distMatrix[i * totalNodes + j];
                    }
                }
            }
            s_r[i] = r_val; // Cache directly in shared memory
            d_r[i] = r_val; // Write-through to global memory
        }
        __syncthreads(); 

        // ---------------------------------------------------------
        // Step 2: Calculate Q-matrix and find the local minimums
        // ---------------------------------------------------------
        float local_minQ = FLT_MAX;
        int local_min_i = -1, local_min_j = -1;

        for (int i = tid; i < nextNodeId; i += blockDim.x) {
            if (!s_active[i]) continue;
            for (int j = i + 1; j < nextNodeId; ++j) {
                if (!s_active[j]) continue;

                float dist_ij = d_distMatrix[i * totalNodes + j];
                // Read r values instantly from L1 shared memory cache
                float q = (currentNodesCount - 2) * dist_ij - s_r[i] - s_r[j];

                if (q < local_minQ) {
                    local_minQ = q;
                    local_min_i = i;
                    local_min_j = j;
                }

                else if (q == local_minQ) {
                    if (i < local_min_i || (i == local_min_i && j < local_min_j)) {
                        local_min_i = i;
                        local_min_j = j;
                    }
                }
            }
        }

        minQ_vals[tid] = local_minQ;
        minQ_i[tid] = local_min_i;
        minQ_j[tid] = local_min_j;
        __syncthreads();

        // ---------------------------------------------------------
        // Step 3: O(log N) Parallel Tree Reduction for Global Minimum
        // ---------------------------------------------------------
        // O(log N) Parallel Tree Reduction with Lexicographical Tie-Breaking
        #pragma unroll
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float q_stride = minQ_vals[tid + stride];
                float q_curr = minQ_vals[tid];
                
                if (q_stride < q_curr) {
                    minQ_vals[tid] = q_stride;
                    minQ_i[tid] = minQ_i[tid + stride];
                    minQ_j[tid] = minQ_j[tid + stride];
                } 
                // EXPLICIT TIE-BREAKER FOR REDUCTION
                else if (q_stride == q_curr) {
                    int i_stride = minQ_i[tid + stride];
                    int j_stride = minQ_j[tid + stride];
                    int i_curr = minQ_i[tid];
                    int j_curr = minQ_j[tid];
                    
                    if (i_stride < i_curr || (i_stride == i_curr && j_stride < j_curr)) {
                        minQ_i[tid] = i_stride;
                        minQ_j[tid] = j_stride;
                    }
                }
            }
            __syncthreads();
        }

        // Thread 0 now effortlessly holds the ultimate winner
        int min_i = minQ_i[0];
        int min_j = minQ_j[0];

        // ---------------------------------------------------------
        // Step 4: Compute Branch Lengths (Thread 0 only)
        // ---------------------------------------------------------
        if (tid == 0) {
            float dist_ij = d_distMatrix[min_i * totalNodes + min_j];
            float dist_i_u = 0.5f * dist_ij + (s_r[min_i] - s_r[min_j]) / (2.0f * (currentNodesCount - 2));
            float dist_j_u = dist_ij - dist_i_u;

            d_left_child[nextNodeId] = min_i;
            d_right_child[nextNodeId] = min_j;
            d_dist_left[nextNodeId] = dist_i_u;
            d_dist_right[nextNodeId] = dist_j_u;

            // Update both shared cache and global mask
            s_active[min_i] = 0;
            s_active[min_j] = 0;
            s_active[nextNodeId] = 1;
            
            d_active[min_i] = false;
            d_active[min_j] = false;
            d_active[nextNodeId] = true;
        }
        __syncthreads(); 

        // ---------------------------------------------------------
        // Step 5: Update Distance Matrix for the new node in Parallel
        // ---------------------------------------------------------
        float dist_ij = d_distMatrix[min_i * totalNodes + min_j];
        
        for (int k = tid; k < nextNodeId; k += blockDim.x) {
            // Note: We skip nextNodeId because we are actively writing its row/col
            if (s_active[k] && k != min_i && k != min_j && k != nextNodeId) {
                float dist_ik = d_distMatrix[min_i * totalNodes + k];
                float dist_jk = d_distMatrix[min_j * totalNodes + k];
                float dist_uk = 0.5f * (dist_ik + dist_jk - dist_ij);

                d_distMatrix[nextNodeId * totalNodes + k] = dist_uk;
                d_distMatrix[k * totalNodes + nextNodeId] = dist_uk;
            }
        }
        __syncthreads(); 

        currentNodesCount--;
        nextNodeId++;
    }
}

// ============================================================================
// Caller Function
// ============================================================================
void GpuAligner::NeighborJoiningCaller(
    int numSeqs, 
    int numPairs, 
    const int* d_pairA_idx, 
    const int* d_pairB_idx, 
    const float* d_out_D,   
    int* h_left_child,      
    int* h_right_child,     
    float* h_dist_left,     
    float* h_dist_right)    
{
    int totalNodes = 2 * numSeqs - 1;
    
    size_t matrixSize = totalNodes * totalNodes * sizeof(float);
    size_t floatArraySize = totalNodes * sizeof(float);
    size_t intArraySize = totalNodes * sizeof(int);
    size_t boolArraySize = totalNodes * sizeof(bool);

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

    cudaMemset(d_distMatrix, 0, matrixSize);
    cudaMemset(d_left_child, -1, intArraySize);
    cudaMemset(d_right_child, -1, intArraySize);
    cudaMemset(d_dist_left, 0, floatArraySize);
    cudaMemset(d_dist_right, 0, floatArraySize);

    // 1. Launch Massively Parallel Setup Kernel
    int maxThreadsNeeded = (numPairs > totalNodes) ? numPairs : totalNodes;
    int setupBlockSize = 256;
    int numBlocks = (maxThreadsNeeded + setupBlockSize - 1) / setupBlockSize;

    InitNJMatrixKernel<<<numBlocks, setupBlockSize>>>(
        numSeqs, numPairs, d_pairA_idx, d_pairB_idx, d_out_D, d_distMatrix, d_active
    );
    cudaDeviceSynchronize(); 

    // 2. Launch Highly Optimized Parallel NJ Kernel
    // MUST BE A POWER OF 2 (e.g., 256, 512) for the parallel reduction to work!
    int njBlockSize = 512; 
    
    // Calculate memory needed for dynamic shared memory (r array + active array)
    size_t dynamicSharedMemSize = (totalNodes * sizeof(float)) + (totalNodes * sizeof(int));

    OptimizedNeighborJoiningKernel<<<1, njBlockSize, dynamicSharedMemSize>>>(
        d_distMatrix, d_active, d_r, 
        d_left_child, d_right_child, 
        d_dist_left, d_dist_right, 
        numSeqs
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Parallel NJ kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); 

    // 3. Copy Topology Back to Host
    cudaMemcpy(h_left_child, d_left_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right_child, d_right_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_left, d_dist_left, floatArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_right, d_dist_right, floatArraySize, cudaMemcpyDeviceToHost);

    // 4. Cleanup
    cudaFree(d_distMatrix);
    cudaFree(d_active);
    cudaFree(d_r);
    cudaFree(d_left_child);
    cudaFree(d_right_child);
    cudaFree(d_dist_left);
    cudaFree(d_dist_right);
}