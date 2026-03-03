#include "masterHeader.cuh"
#include <iostream>
#include <cstdint>
#include <cstring>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>

#define HASH_SEED 1337
// Helper function to rotate 64-bits left
__device__ inline uint64_t rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

// 64-bit final mixing function (Avalanche)
__device__ inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

// MurmurHash3 (x64_128 variant) modified to output 64 bits
__device__ uint64_t MurmurHash3_x64_64(const void* key, int len, uint64_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 16; 

    uint64_t h1 = seed;
    uint64_t h2 = seed;
    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    const uint64_t* blocks = (const uint64_t*)(data);

    for (int i = 0; i < nblocks; i++) {
        uint64_t k1 = blocks[i * 2 + 0];
        uint64_t k2 = blocks[i * 2 + 1];

        k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
        h1 = rotl64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729ULL;

        k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
        h2 = rotl64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5ULL;
    }

    const uint8_t* tail = (const uint8_t*)(data + nblocks * 16);
    uint64_t k1 = 0;
    uint64_t k2 = 0;

    switch (len & 15) {
        case 15: k2 ^= ((uint64_t)tail[14]) << 48;
        case 14: k2 ^= ((uint64_t)tail[13]) << 40;
        case 13: k2 ^= ((uint64_t)tail[12]) << 32;
        case 12: k2 ^= ((uint64_t)tail[11]) << 24;
        case 11: k2 ^= ((uint64_t)tail[10]) << 16;
        case 10: k2 ^= ((uint64_t)tail[ 9]) << 8;
        case  9: k2 ^= ((uint64_t)tail[ 8]) << 0;
                 k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
        case  8: k1 ^= ((uint64_t)tail[ 7]) << 56;
        case  7: k1 ^= ((uint64_t)tail[ 6]) << 48;
        case  6: k1 ^= ((uint64_t)tail[ 5]) << 40;
        case  5: k1 ^= ((uint64_t)tail[ 4]) << 32;
        case  4: k1 ^= ((uint64_t)tail[ 3]) << 24;
        case  3: k1 ^= ((uint64_t)tail[ 2]) << 16;
        case  2: k1 ^= ((uint64_t)tail[ 1]) << 8;
        case  1: k1 ^= ((uint64_t)tail[ 0]) << 0;
                 k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
    };

    h1 ^= len; h2 ^= len;
    h1 += h2;  h2 += h1;
    h1 = fmix64(h1); h2 = fmix64(h2);
    h1 += h2;  h2 += h1;

    return h1;
}

__global__ void HashAllKmersKernel(const uint64_t* d_kmerArr, uint64_t* d_hashArr, uint32_t numKmers, int kmerByteLen) {
    // Standard 1D grid stride
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numKmers) {
        // Read the k-mer
        uint64_t kmer = d_kmerArr[tid]; 
        
        // Hash it and store in the parallel hash array
        d_hashArr[tid] = MurmurHash3_x64_64(&kmer, kmerByteLen, HASH_SEED);
    }
}

void GpuAligner::MurmurHashCaller(uint32_t numKmers, int kmerByteLen, uint32_t bottomK, uint64_t* hOut_sketch) {
    int numBlocks = 1024; 
    int blockSize = 512; 

    // Clear previous errors
    cudaGetLastError(); 

    // Launch Kernel
    HashAllKmersKernel<<<numBlocks, blockSize>>>(d_kmerArr, d_hashArr, numKmers, kmerByteLen);

    // Error Checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Kernel launch failed: %s!\n", cudaGetErrorString(err));
    }
    
    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: Device sync failed: %s!\n", cudaGetErrorString(err));
    }

    // Sort the hashed array on the GPU to find the smallest values (Bottom-k)
    // Thrust perfectly maps to CUDA's execution policy and is highly optimized.
    thrust::device_ptr<uint64_t> dev_hash_ptr(d_hashArr);
    thrust::sort(thrust::device, dev_hash_ptr, dev_hash_ptr + numKmers);

    // 4. Extract the bottom 'K' hashes
    // Ensure we don't try to copy more k-mers than actually exist in small sequences
    uint32_t elementsToCopy = (numKmers < bottomK) ? numKmers : bottomK;
    
    // Copy the sketch (smallest hashes) back to host memory (or wherever hOut_sketch lives)
    err = cudaMemcpy(hOut_sketch, d_hashArr, elementsToCopy * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        printf("ERROR: Memcpy failed: %s!\n", cudaGetErrorString(err));
    }

}
