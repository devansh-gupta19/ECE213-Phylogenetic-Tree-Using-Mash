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

    int bs = blockDim.x;
    int gs = gridDim.x;

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    uint32_t k = kmerSize;
    // Safely cast to 64-bit to prevent bitwise shift overflow
    uint64_t mask = (1ULL << (2*k))-1; 

    uint32_t start_idx = bx * bs + tx;
    // Running sequentially on a single thread
    for (uint32_t i = start_idx; i < numKmers; i += bs*gs) { 
        uint32_t bit_offset = i * 2;
        uint32_t index = bit_offset / 32;
        uint32_t shift = bit_offset % 32;

        uint64_t kmer = 0;
        
        // Cast to uint64_t BEFORE bitwise operations
        uint64_t w0 = d_compressedSeq[index];
        uint64_t w1 = d_compressedSeq[index + 1];

        if (shift <= (64 - 2 * kmerSize)) {
            // Fits within two 32-bit words
            kmer = (w0 >> shift) | (w1 << (32 - shift));
        } else {
            // Spans across three 32-bit words
            uint64_t w2 = d_compressedSeq[index + 2];
            kmer = (w0 >> shift) | (w1 << (32 - shift)) | (w2 << (64 - shift));
        }

        // Store ONLY the kmer value masked to the correct bit length
        d_kmerArr[i] = kmer & mask;
    }
}



/**
 * Constructs seed table, consisting and kmerPos arrrays
 * on the GPU.
*/
uint32_t GpuAligner::seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* kmerArr) {

    // TODO: make sure to appropriately set the values below
    int numBlocks = 1024; // i.e. number of thread blocks on the GPU
    int blockSize = 512; // i.e. number of GPU threads per thread block

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

    // TODO: Eventually remove this function, no need to transfer kmers to host
    transferData2Host(kmerArr, numKmers);


    // Wait for all computation on GPU device to finish. Needed to ensure
    // correct runtime profiling results for this function.
    cudaDeviceSynchronize();

    return numUniqueKmers;
}