#include <cstdint>  // Fixes uint32_t
#include <cstddef>  // Fixes size_t
#include <cstdio>

// C++ compilers don't understand __global__, so we hide it if not using NVCC
#ifndef __CUDACC__
#define __global__
#endif

struct GpuAligner {
    //std::vector<Sequence> seqs;
    int32_t longestLen;
    int32_t numPairs;

    char*    d_seqs;
    uint8_t* d_tb;
    int32_t* d_seqLen;
    int32_t* d_info;

    void seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,
    uint32_t kmerSize,
    size_t* kmerArr);

    void MurmurHashCaller(const void* key, int len, uint32_t seed, uint32_t *hOut);

};

void printGpuProperties();