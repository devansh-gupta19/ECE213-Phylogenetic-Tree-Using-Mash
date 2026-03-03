#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>


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
    uint64_t *d_hashArr;

    // MASH Pairwise pointers
    uint64_t* d_allSketches;
    int* d_pairA_idx;
    int* d_pairB_idx;
    float* d_out_J;
    float* d_out_D;
    float* d_out_P;


    void allocateMem(int32_t len, int32_t numKmers, int32_t kmerSize);
    void transferData2Device(uint32_t *seq, int32_t len, int32_t kmerSize);
    void transferData2Host(size_t *kmerArr, int32_t kmerArrSize);
    void clearAndReset ();
    uint32_t seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    size_t* kmerArr);

    void MurmurHashCaller(uint32_t numKmers, int kmerByteLen, uint32_t bottomK, uint64_t* hOut_sketch);

    // Mash calculation
    void allocateMashMem(int totalSketchElements, int numPairs);
    void transferMashDataToDevice(const uint64_t* h_flatSketches, const int* h_pairA_idx, const int* h_pairB_idx, int totalSketchElements, int numPairs);
    void transferMashResultsToHost(float* h_out_J, float* h_out_D, float* h_out_P, int numPairs);
    void freeMashMem();
    
    // Mash calculation
    void MashDistanceCalculationCaller(
        int numPairs, 
        int sketchSize, 
        int kmerSize, 
        int numSeqs,           // NEW
        int* h_left_child,     // NEW
        int* h_right_child,    // NEW
        float* h_dist_left,    // NEW
        float* h_dist_right    // NEW
    );

    // Neighbor Joining
    void NeighborJoiningCaller(
        int numSeqs, 
        int numPairs, 
        const int* d_pairA_idx, 
        const int* d_pairB_idx, 
        const float* d_out_D,
        int* h_left_child,     // NEW
        int* h_right_child,    // NEW
        float* h_dist_left,    // NEW
        float* h_dist_right    // NEW
    );
};

void printGpuProperties();

