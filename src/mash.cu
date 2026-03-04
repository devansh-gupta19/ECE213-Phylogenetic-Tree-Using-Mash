#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "masterHeader.cuh"

// ============================================================================
// 1. HIGHLY OPTIMIZED MASH DISTANCE KERNEL
// ============================================================================
__global__ void ComputeMashDistancesKernel(
    const uint64_t* __restrict__ d_allSketches, 
    const int* __restrict__ d_pairA_idx,        
    const int* __restrict__ d_pairB_idx,        
    float* __restrict__ d_out_J, 
    float* __restrict__ d_out_D, 
    float* __restrict__ d_out_P,
    int numPairs,
    int sketchSize, 
    int kmerSize) 
{
    // Global thread ID directly corresponds to a specific pair index
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // --- OPTIMIZATION 1: Shared Memory Lookup Table ---
    // lgammaf() (log-gamma) is an incredibly expensive transcendental function.
    // Instead of computing it millions of times inside the p-value loop, we 
    // collaboratively precompute the log-factorials once per block and cache 
    // them in ultra-fast L1 shared memory.
    extern __shared__ float s_lgamma[];
    
    // Threads work together to build the lookup table
    for (int k = tid; k <= sketchSize; k += blockDim.x) {
        s_lgamma[k] = lgammaf((float)k + 1.0f);
    }
    __syncthreads(); // Wait for the shared cache to finish building

    // If this thread maps outside our valid workload, safely exit
    if (pairIdx >= numPairs) return;

    // Look up which two sequences this specific thread is comparing
    int seqA_id = d_pairA_idx[pairIdx];
    int seqB_id = d_pairB_idx[pairIdx];

    // --- OPTIMIZATION 2: Read-Only Data Caching ---
    // The __restrict__ keyword tells the compiler it's safe to load these 
    // sketches through the fast Read-Only Data Cache.
    const uint64_t* sketchA = &d_allSketches[seqA_id * sketchSize];
    const uint64_t* sketchB = &d_allSketches[seqB_id * sketchSize];

    // (a) Compute |As ∩ Bs| using a two-pointer intersection
    int intersection = 0;
    int i = 0; 
    int j = 0;

    while (i < sketchSize && j < sketchSize) {
        uint64_t a = sketchA[i];
        uint64_t b = sketchB[j];
        if (a == b) {
            intersection++;
            i++;
            j++;
        } else if (a < b) {
            i++;
        } else {
            j++;
        }
    }

    // (b & c) Compute Union and Jaccard Index
    float union_size = (2.0f * (float)sketchSize) - (float)intersection;
    float jaccard = (union_size > 0.0f) ? ((float)intersection / union_size) : 0.0f;
    
    d_out_J[pairIdx] = jaccard;

    // (d) Compute Mash Distance
    float mash_dist = 1.0f; 
    if (jaccard > 0.0f) {
        mash_dist = -1.0f / (float)kmerSize * logf((2.0f * jaccard) / (1.0f + jaccard));
    }
    
    d_out_D[pairIdx] = mash_dist;

    // (e) Compute P-Value Binomial Tail
    float q = powf(0.25f, (float)kmerSize); 
    float log_q = logf(q);
    float log_1_minus_q = logf(1.0f - q);
    float p_val = 0.0f;
    
    // Grab the precomputed N! log value instantly from shared memory
    float log_sketchSize_factorial = s_lgamma[sketchSize];

    // --- OPTIMIZATION 3: O(1) Cache Math ---
    for (int k = intersection; k <= sketchSize; ++k) {
        // We replace three heavy lgammaf() calls with three instant L1 cache reads
        float log_term = log_sketchSize_factorial 
                       - s_lgamma[k] 
                       - s_lgamma[sketchSize - k];
                       
        log_term += ((float)k * log_q) + ((float)(sketchSize - k) * log_1_minus_q);
        p_val += expf(log_term);
    }
    
    d_out_P[pairIdx] = p_val;
}


// ============================================================================
// 2. MASH DISTANCE CALLER
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
    // Launch a massive grid. E.g., for 500,000 pairs, we spawn ~1,954 blocks of 256 threads
    int blockSize = 256;
    int numBlocks = (numPairs + blockSize - 1) / blockSize;

    // We need (sketchSize + 1) floats of shared memory per block for the lgamma table
    size_t dynamicSharedMemSize = (sketchSize + 1) * sizeof(float);

    ComputeMashDistancesKernel<<<numBlocks, blockSize, dynamicSharedMemSize>>>(
        d_allSketches, d_pairA_idx, d_pairB_idx, d_out_J, d_out_D, d_out_P, numPairs, sketchSize, kmerSize);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Mash kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    
    // Chain directly into the Neighbor Joining pipeline
    NeighborJoiningCaller(
        numSeqs, numPairs, 
        d_pairA_idx, d_pairB_idx, d_out_D, 
        h_left_child, h_right_child, h_dist_left, h_dist_right
    );
}