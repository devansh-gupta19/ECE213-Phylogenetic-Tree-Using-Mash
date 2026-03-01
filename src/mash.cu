#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <float.h> // Added this to get FLT_MAX for the NJ algorithm
#include "masterHeader.cuh"

// ============================================================================
// 1. MASH DISTANCE KERNEL
// ============================================================================
__global__ void ComputeMashDistancesKernel(
    const uint64_t* d_allSketches, // Single flattened array of ALL sketches
    const int* d_pairA_idx,        // Sequence A index for this block
    const int* d_pairB_idx,        // Sequence B index for this block
    float* d_out_J, 
    float* d_out_D, 
    float* d_out_P,
    int numPairs,
    int sketchSize, 
    int kmerSize) 
{
    // 1 block handles exactly 1 pair
    // int pairIdx = blockIdx.x;
    int tid = threadIdx.x;

    // TODO: Eventually remove this loop
    for (int pairIdx = 0; pairIdx < numPairs; pairIdx++)
    {
        // Look up which two sequences this block is comparing
        int seqA_id = d_pairA_idx[pairIdx];
        int seqB_id = d_pairB_idx[pairIdx];

        // Point to the correct segments in the flattened sketch array
        const uint64_t* sketchA = &d_allSketches[seqA_id * sketchSize];
        const uint64_t* sketchB = &d_allSketches[seqB_id * sketchSize];

        // TODO (a): Compute |As ∩ Bs|
        int intersection = 0;
        int i = 0; 
        int j = 0;

        // Sequential two-pointer approach for sorted arrays (baseline)
        while (i < sketchSize && j < sketchSize) {
            if (sketchA[i] == sketchB[j]) {
                intersection++;
                i++;
                j++;
            } else if (sketchA[i] < sketchB[j]) {
                i++;
            } else {
                j++;
            }
        }

        // TODO (b): Compute |As ∪ Bs| via |As| + |Bs| - |As ∩ Bs|
        float union_size = (2.0f * (float)sketchSize) - (float)intersection;

        // TODO (c): Compute Jaccard index J = |As ∩ Bs| / |As ∪ Bs|
        float jaccard = 0.0f;
        if (union_size > 0.0f) {
            jaccard = (float)intersection / union_size;
        }
        
        // Write result to the specific index for this pair
        if (tid == 0) d_out_J[pairIdx] = jaccard;

        // TODO (d): Compute Mash distance D
        float mash_dist = 1.0f; // Default value if no similarity
        if (jaccard > 0.0f) {
            // Use CUDA's logf() instead of std::log()
            mash_dist = -1.0f / (float)kmerSize * logf((2.0f * jaccard) / (1.0f + jaccard));
        }
        
        if (tid == 0) d_out_D[pairIdx] = mash_dist;

        // TODO (e): Compute and store p-value
        // Use CUDA's powf() and logf()
        float q = powf(0.25f, (float)kmerSize); // 1 / 4^k
        float log_q = logf(q);
        float log_1_minus_q = logf(1.0f - q);
        
        float p_val = 0.0f;

        // Sequential summation of the binomial tail
        for (int k = intersection; k <= sketchSize; ++k) {
            // Use CUDA's lgammaf() instead of std::lgamma()
            float log_term = lgammaf((float)sketchSize + 1.0f) 
                        - lgammaf((float)k + 1.0f) 
                        - lgammaf((float)(sketchSize - k) + 1.0f);
                        
            log_term += ((float)k * log_q) + ((float)(sketchSize - k) * log_1_minus_q);
            
            // Use CUDA's expf()
            p_val += expf(log_term);
        }
        
        if (tid == 0) d_out_P[pairIdx] = p_val;
    }
}

// ============================================================================
// 2. NEIGHBOR JOINING SETUP KERNEL (NEW)
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

    // Populate the 2D symmetric distance matrix from the 1D arrays
    if (tid < numPairs) {
        int r = d_pairA_idx[tid];
        int c = d_pairB_idx[tid];
        float dist = d_out_D[tid];
        
        d_distMatrix[r * totalNodes + c] = dist;
        d_distMatrix[c * totalNodes + r] = dist; // Symmetric
    }

    // Initialize the active mask
    if (tid < totalNodes) {
        d_active[tid] = (tid < numSeqs);
    }
}

// ============================================================================
// 3. SEQUENTIAL NEIGHBOR JOINING KERNEL (NEW)
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
        // Calculate Net Divergence r[i]
        for (int i = 0; i < nextNodeId; ++i) {
            d_r[i] = 0.0f;
            if (!d_active[i]) continue;
            
            for (int j = 0; j < nextNodeId; ++j) {
                if (d_active[j] && i != j) {
                    d_r[i] += d_distMatrix[i * totalNodes + j];
                }
            }
        }

        // Calculate Q-matrix and find the minimum pair
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

        // Compute Branch Lengths and Update Tree Topology
        float dist_ij = d_distMatrix[min_i * totalNodes + min_j];
        float dist_i_u = 0.5f * dist_ij + (d_r[min_i] - d_r[min_j]) / (2.0f * (currentNodesCount - 2));
        float dist_j_u = dist_ij - dist_i_u;

        d_left_child[nextNodeId] = min_i;
        d_right_child[nextNodeId] = min_j;
        d_dist_left[nextNodeId] = dist_i_u;
        d_dist_right[nextNodeId] = dist_j_u;

        // Update the Distance Matrix for the new node
        for (int k = 0; k < nextNodeId; ++k) {
            if (d_active[k] && k != min_i && k != min_j) {
                float dist_ik = d_distMatrix[min_i * totalNodes + k];
                float dist_jk = d_distMatrix[min_j * totalNodes + k];
                float dist_uk = 0.5f * (dist_ik + dist_jk - dist_ij);

                d_distMatrix[nextNodeId * totalNodes + k] = dist_uk;
                d_distMatrix[k * totalNodes + nextNodeId] = dist_uk;
            }
        }

        // Update Active Masks
        d_active[min_i] = false;
        d_active[min_j] = false;
        d_active[nextNodeId] = true;

        currentNodesCount--;
        nextNodeId++;
    }
}

// ============================================================================
// 4. NEIGHBOR JOINING CALLER (NEW)
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

    int maxThreadsNeeded = (numPairs > totalNodes) ? numPairs : totalNodes;
    int blockSize = 256;
    int numBlocks = (maxThreadsNeeded + blockSize - 1) / blockSize;

    InitNJMatrixKernel<<<numBlocks, blockSize>>>(
        numSeqs, numPairs, d_pairA_idx, d_pairB_idx, d_out_D, d_distMatrix, d_active
    );
    cudaDeviceSynchronize(); 

    SequentialNeighborJoiningKernel<<<1, 1>>>(
        d_distMatrix, d_active, d_r, 
        d_left_child, d_right_child, 
        d_dist_left, d_dist_right, 
        numSeqs
    );
    cudaDeviceSynchronize(); 

    cudaMemcpy(h_left_child, d_left_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_right_child, d_right_child, intArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_left, d_dist_left, floatArraySize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dist_right, d_dist_right, floatArraySize, cudaMemcpyDeviceToHost);

    cudaFree(d_distMatrix);
    cudaFree(d_active);
    cudaFree(d_r);
    cudaFree(d_left_child);
    cudaFree(d_right_child);
    cudaFree(d_dist_left);
    cudaFree(d_dist_right);
}

// ============================================================================
// 5. MASH DISTANCE CALLER
// ============================================================================
void GpuAligner::MashDistanceCalculationCaller(
    int numPairs, 
    int sketchSize, 
    int kmerSize, 
    int numSeqs,
    int* h_left_child,   
    int* h_right_child,  
    float* h_dist_left,  
    float* h_dist_right) 
{
    // Launch exactly 1 block for every pair you need to process
    int numBlocks = 1;  //numPairs; 
    int blockSize = 1;

    ComputeMashDistancesKernel<<<numBlocks, blockSize>>>(
        d_allSketches, d_pairA_idx, d_pairB_idx, d_out_J, d_out_D, d_out_P, numPairs, sketchSize, kmerSize);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Mash kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    
    // Pass the host pointers into the NJ caller so the data reaches the CPU
    NeighborJoiningCaller(
        numSeqs, numPairs, 
        d_pairA_idx, d_pairB_idx, d_out_D, 
        h_left_child, h_right_child, h_dist_left, h_dist_right
    );
}