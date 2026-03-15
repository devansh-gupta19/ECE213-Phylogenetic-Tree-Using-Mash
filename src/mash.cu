#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "masterHeader.cuh"

// MASH Distance Kernel
__global__ void ComputeMashDistancesKernel(
    const uint64_t* __restrict__ d_allSketches, 
    const int* __restrict__ d_pairA_idx,        
    const int* __restrict__ d_pairB_idx,        
    float* __restrict__ d_out_J, 
    float* __restrict__ d_out_D, 
    int numPairs,
    int sketchSize, 
    int kmerSize) 
{
    // Global thread ID directly corresponds to a specific pair index
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // If this thread maps outside our valid workload, safely exit
    if (pairIdx >= numPairs) return;

    // Look up which two sequences this specific thread is comparing
    int seqA_id = d_pairA_idx[pairIdx];
    int seqB_id = d_pairB_idx[pairIdx];

    // Read-Only Data Caching
    // The __restrict__ keyword tells the compiler it's safe to load these 
    // sketches through the fast Read-Only Data Cache.
    const uint64_t* sketchA = &d_allSketches[seqA_id * sketchSize];
    const uint64_t* sketchB = &d_allSketches[seqB_id * sketchSize];

    // Compute |As ∩ Bs| using a two-pointer intersection
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

    // Compute Union and Jaccard Index
    float union_size = (2.0f * (float)sketchSize) - (float)intersection;
    float jaccard = (union_size > 0.0f) ? ((float)intersection / union_size) : 0.0f;
    
    d_out_J[pairIdx] = jaccard;

    // Compute Mash Distance
    float mash_dist = 1.0f; 
    if (jaccard > 0.0f) {
        mash_dist = -1.0f / (float)kmerSize * logf((2.0f * jaccard) / (1.0f + jaccard));
    }
    
    d_out_D[pairIdx] = mash_dist;
}


// MASH Distance caller function
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
        d_allSketches, d_pairA_idx, d_pairB_idx, d_out_J, d_out_D, numPairs, sketchSize, kmerSize);

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