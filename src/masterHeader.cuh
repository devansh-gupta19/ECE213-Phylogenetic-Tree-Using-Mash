#include <cstdint>  // Fixes uint32_t
#include <cstddef>  // Fixes size_t
#include <cstdio>

// C++ compilers don't understand __global__, so we hide it if not using NVCC
#ifndef __CUDACC__
#define __global__
#endif

struct GpuAligner {
    int32_t longestLen;
    //int32_t numPairs;

    //char*    d_seqs;
    uint32_t*    d_seqs;
    int32_t* d_seqLen;
    int32_t *d_kmerSize;
    size_t *d_kmerArr;
    // int32_t* d_info;

    void seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,
    uint32_t kmerSize,
    size_t* kmerArr);

    void allocateMem(int32_t len, int32_t numKmers, int32_t kmerSize);
    void transferData2Device(uint32_t *seq, int32_t len, int32_t kmerSize);
    void transferData2Host(size_t *kmerArr, int32_t kmerArrSize);

    void seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* kmerArr);

    void MurmurHashCaller(const void* key, int len, uint32_t seed, uint32_t *hOut);

};

void printGpuProperties();