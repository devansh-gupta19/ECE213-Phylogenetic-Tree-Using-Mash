#include "masterHeader.cuh"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>

/**
 * Finds kmers for the compressed sequence creates an array with elements
 * containing the 64-bit concatenated value consisting of the kmer value in the
 * first 32 bits and the kmer position in the last 32 bits. The values are
 * stored in the arrary kmerPos, with i-th element corresponding to the i-th
 * kmer in the sequence
 *
 * TODO: parallelize this function
 */
__global__ void kmerArrCreate(
    uint32_t* d_compressedSeq,
    uint32_t numKmers, 
    uint32_t kmerSize,
    size_t* d_kmerArr) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    uint32_t k = kmerSize;
    // Safely cast to 64-bit to prevent bitwise shift overflow
    uint64_t mask = (1ULL << (2*k))-1; 
    size_t kmer = 0;

    // Running sequentially on a single thread
    if ((bx == 0) && (tx == 0)) {
        for (uint32_t i = 0; i < numKmers; i++) { 
            uint32_t index = i/16;
            uint32_t shift1 = 2*(i%16);
            if (shift1 > 0) {
                uint32_t shift2 = 32-shift1;
                kmer = ((d_compressedSeq[index] >> shift1) | (d_compressedSeq[index+1] << shift2)) & mask;
            } else {
                kmer = d_compressedSeq[index] & mask;
            }

            // Store ONLY the kmer value. No position concatenated.
            d_kmerArr[i] = kmer; 
        }
    }
}



/**
 * Constructs seed table, consisting and kmerPos arrrays
 * on the GPU.
*/
void GpuAligner::seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* kmerArr) {

    // TODO: make sure to appropriately set the values below
    int numBlocks = 1; // i.e. number of thread blocks on the GPU
    int blockSize = 1; // i.e. number of GPU threads per thread block

    std::string berr = cudaGetErrorString(cudaGetLastError());
    if (berr != "no error") printf("ERROR: Before kernel %s!\n", berr.c_str());


    transferData2Device(compressedSeq, seqLen, kmerSize);
    
    kmerArrCreate<<<numBlocks, blockSize>>>(d_seqs, numKmers, kmerSize, d_kmerArr);

    std::string aerr = cudaGetErrorString(cudaGetLastError());
    if (aerr != "no error") printf("ERROR: After kernel %s!\n", aerr.c_str());
    // Parallel sort the kmerPos array on the GPU device using the thrust
    // library (https://thrust.github.io/)
    thrust::device_ptr<size_t> kmerArrPtr(d_kmerArr);
    thrust::sort(kmerArrPtr, kmerArrPtr + numKmers);

    thrust::device_ptr<size_t> newEnd = thrust::unique(kmerArrPtr, kmerArrPtr + numKmers);

    // Calculate the new size of the array containing only unique k-mers
    size_t numUniqueKmers = newEnd - kmerArrPtr;


    transferData2Host(kmerArr, numKmers);


    // Wait for all computation on GPU device to finish. Needed to ensure
    // correct runtime profiling results for this function.
    cudaDeviceSynchronize();
}