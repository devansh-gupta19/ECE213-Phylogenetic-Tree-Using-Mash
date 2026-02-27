#include "masterHeader.cuh"
#include <iostream>
#include <cstdint>
#include <cstring>

// Helper function to rotate bits left
__device__ uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

// Final mixing function (The Avalanche / fmix step)
__device__ uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

// Main MurmurHash3 function (32-bit version)
__global__ void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, uint32_t *hOut) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;

    // --- 1. Initialization ---
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    // --- 2. The Body (Block Processing) ---
    // Read the data in 4-byte chunks
    const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);

    for (int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];

        // Scramble the block
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        // Mix the scrambled block into the hash state
        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    // --- 3. The Tail (Remainder Processing) ---
    // Handle the remaining bytes (1, 2, or 3 bytes)
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
    uint32_t k1 = 0;

    // Fall-through switch statement to pack leftover bytes
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; 
                k1 = rotl32(k1, 15); 
                k1 *= c2; 
                h1 ^= k1;
    };

    // --- 4. Finalization & Avalanche ---
    // XOR in the total length of the key
    h1 ^= len;

    // Force all bits to avalanche
    *hOut = fmix32(h1);
}


void GpuAligner::MurmurHashCaller(const void* key, int len, uint32_t seed, uint32_t *hOut) {
    int numBlocks = 1; // i.e. number of thread blocks on the GPU
    int blockSize = 1; // i.e. number of GPU threads per thread block

    std::string berr = cudaGetErrorString(cudaGetLastError());
    if (berr != "no error") printf("ERROR: Before kernel %s!\n", berr.c_str());
    
    MurmurHash3_x86_32<<<numBlocks, blockSize>>>(key, len, seed, hOut);

    std::string aerr = cudaGetErrorString(cudaGetLastError());
    if (aerr != "no error") printf("ERROR: After kernel %s!\n", aerr.c_str());
    // Wait for all computation on GPU device to finish. Needed to ensure
    // correct runtime profiling results for this function.
    cudaDeviceSynchronize();
}

