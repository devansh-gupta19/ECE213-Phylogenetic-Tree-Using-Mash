#include <cstdint>
#include <stdio.h>
#include <math.h>
#include "masterHeader.cuh"

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

        // ========================================================================
        // TODO (a): Compute |As ∩ Bs|
        // Map parallelism across threads in a block. 
        // ========================================================================
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

        // ========================================================================
        // TODO (b): Compute |As ∪ Bs| via |As| + |Bs| - |As ∩ Bs|
        // Map parallelism across threads in a block.
        // ========================================================================
        float union_size = (2.0f * (float)sketchSize) - (float)intersection;

        // ========================================================================
        // TODO (c): Compute Jaccard index J = |As ∩ Bs| / |As ∪ Bs|
        // Map parallelism across threads in a block.
        // ========================================================================
        float jaccard = 0.0f;
        if (union_size > 0.0f) {
            jaccard = (float)intersection / union_size;
        }
        
        // Write result to the specific index for this pair
        if (tid == 0) d_out_J[pairIdx] = jaccard;

        // ========================================================================
        // TODO (d): Compute Mash distance D
        // Map parallelism across threads in a block.
        // ========================================================================
        float mash_dist = 1.0f; // Default value if no similarity
        if (jaccard > 0.0f) {
            // Use CUDA's logf() instead of std::log()
            mash_dist = -1.0f / (float)kmerSize * logf((2.0f * jaccard) / (1.0f + jaccard));
        }
        
        if (tid == 0) d_out_D[pairIdx] = mash_dist;

        // ========================================================================
        // TODO (e): Compute and store p-value
        // Map parallelism across threads in a block.
        // ========================================================================
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

void GpuAligner::MashDistanceCalculationCaller(int numPairs, int sketchSize, int kmerSize) 
{
    // Launch exactly 1 block for every pair you need to process
    int numBlocks = 1;  //numPairs; 
    // You will increase this when you parallelize the TODO sections.
    int blockSize = 1;

    ComputeMashDistancesKernel<<<numBlocks, blockSize>>>(
        d_allSketches, d_pairA_idx, d_pairB_idx, d_out_J, d_out_D, d_out_P, numPairs, sketchSize, kmerSize);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Mash kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
