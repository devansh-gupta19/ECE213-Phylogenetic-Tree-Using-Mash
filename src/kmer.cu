#include "masterHeader.cuh"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_kmerArr) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // HINT: Values below could be useful for parallelizing the code
    //int bs = blockDim.x;
    //int gs = gridDim.x;

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    // Helps mask the non kmer bits from compressed sequence. E.g. for k=2,
    // mask=0x1111 and for k=3, mask=0x111111
    uint32_t mask = (1 << 2*k)-1;
    size_t kmer = 0;

    // HINT: the if statement below ensures only the first thread of the first
    // block does all the computation. This statement might have to be removed
    // during parallelization
    if ((bx == 0) && (tx == 0)) {
        for (uint32_t i = 0; i <= N-k; i++) {
            uint32_t index = i/16;
            uint32_t shift1 = 2*(i%16);
            if (shift1 > 0) {
                uint32_t shift2 = 32-shift1;
                kmer = ((d_compressedSeq[index] >> shift1) | (d_compressedSeq[index+1] << shift2)) & mask;
            } else {
                kmer = d_compressedSeq[index] & mask;
            }

            // Concatenate kmer value (first 32-bits) with its position (last
            // 32-bits)
            d_kmerArr[i] = kmer;
        }
    }
}


/**
 * Constructs seed table, consisting of kmerOffset and kmerPos arrrays
 * on the GPU.
*/
void seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,

    uint32_t kmerSize,

    uint32_t* kmerOffset,
    size_t* kmerArr) {

    // TODO: make sure to appropriately set the values below
    int numBlocks = 1; // i.e. number of thread blocks on the GPU
    int blockSize = 1; // i.e. number of GPU threads per thread block

    std::string berr = cudaGetErrorString(cudaGetLastError());
    if (berr != "no error") printf("ERROR: Before kernel %s!\n", berr.c_str());
    
    kmerArrCreate<<<numBlocks, blockSize>>>(compressedSeq, seqLen, kmerSize, kmerArr);

    std::string aerr = cudaGetErrorString(cudaGetLastError());
    if (aerr != "no error") printf("ERROR: After kernel %s!\n", aerr.c_str());
    // Parallel sort the kmerPos array on the GPU device using the thrust
    // library (https://thrust.github.io/)
    thrust::device_ptr<size_t> kmerArrPtr(kmerArr);
    thrust::sort(kmerArrPtr, kmerArrPtr+seqLen-kmerSize+1);

    thrust::device_ptr<size_t> newEnd = thrust::unique(kmerArrPtr, kmerArrPtr + seqLen - kmerSize + 1);

    // Calculate the new size of the array containing only unique k-mers
    size_t numUniqueKmers = newEnd - kmerArrPtr;

    // Wait for all computation on GPU device to finish. Needed to ensure
    // correct runtime profiling results for this function.
    cudaDeviceSynchronize();
}