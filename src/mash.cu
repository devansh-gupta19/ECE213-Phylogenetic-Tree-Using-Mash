#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "masterHeader.cuh"

// MASH Distance Kernel
__global__ void ComputeMashDistancesKernel(
    const uint64_t* d_allSketches, // Single flattened array of ALL sketches
    const int* d_pairA_idx,        // Sequence A index for this block
    const int* d_pairB_idx,        // Sequence B index for this block
    float* d_out_J, 
    float* d_out_D, 
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

        // Compute |As ∩ Bs|
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

        // Compute |As ∪ Bs| via |As| + |Bs| - |As ∩ Bs|
        float union_size = (2.0f * (float)sketchSize) - (float)intersection;

        // Compute Jaccard index J = |As ∩ Bs| / |As ∪ Bs|
        float jaccard = 0.0f;
        if (union_size > 0.0f) {
            jaccard = (float)intersection / union_size;
        }
        
        // Write result to the specific index for this pair
        if (tid == 0) d_out_J[pairIdx] = jaccard;

        // Compute Mash distance D
        float mash_dist = 1.0f; // Default value if no similarity
        if (jaccard > 0.0f) {
            mash_dist = -1.0f / (float)kmerSize * logf((2.0f * jaccard) / (1.0f + jaccard));
        }
        if (tid == 0) d_out_D[pairIdx] = mash_dist;
    }
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
    // Launch exactly 1 block for every pair you need to process
    int numBlocks = 1;  //numPairs;
    int blockSize = 1;

    ComputeMashDistancesKernel<<<numBlocks, blockSize>>>(
        d_allSketches, d_pairA_idx, d_pairB_idx, d_out_J, d_out_D, numPairs, sketchSize, kmerSize);

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
