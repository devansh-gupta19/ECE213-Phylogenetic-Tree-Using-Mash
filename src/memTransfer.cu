#include "masterHeader.cuh"

// Macro for catching CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "GPU_ERROR: %s:%d: %s (%s)\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err), cudaGetErrorName(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Allocates GPU memory for sequences, lengths, and traceback paths.
 * Calculates 'longestLen' to determine the stride for flattening the sequence array.
 */
void GpuAligner::allocateMem(int32_t len, int32_t numKmers, int32_t kmerSize) {

    // 1. Allocate flat array for all sequences (Reference + Query pairs)
    CUDA_CHECK(cudaMalloc(&d_seqs, (len+2) * sizeof(uint32_t)));
    // Zero out the memory so the padded integer is safely 0
    CUDA_CHECK(cudaMemset(d_seqs, 0, (len + 2) * sizeof(uint32_t)));

    // 2. Allocate array for sequence lengths (to handle padding correctly)
    CUDA_CHECK(cudaMalloc(&d_kmerArr, numKmers * sizeof(size_t)));

    CUDA_CHECK(cudaMalloc(&d_hashArr, numKmers * sizeof(uint64_t)));

}

/**
 * Flattens the host sequence objects into a single 1D array and transfers to GPU.
 */
void GpuAligner::transferData2Device(uint32_t *seq, int32_t len, int32_t kmerSize) {

    // 2. Transfer flattened sequences to Device
    CUDA_CHECK(cudaMemcpy(d_seqs, seq, len * sizeof(uint32_t), cudaMemcpyHostToDevice));

}

/**
 * Copies the computed traceback paths from GPU back to Host.
 */
void GpuAligner::transferData2Host(size_t *kmerArr, int32_t kmerArrSize) {

    CUDA_CHECK(cudaMemcpy(kmerArr, d_kmerArr, kmerArrSize * sizeof(size_t), cudaMemcpyDeviceToHost));
}


void GpuAligner::clearAndReset () {
    cudaFree(d_seqs);
    cudaFree(d_kmerArr);
    cudaFree(d_hashArr);
}