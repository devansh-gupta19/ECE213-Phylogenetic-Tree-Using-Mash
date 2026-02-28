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


void GpuAligner::allocateMashMem(int totalSketchElements, int numPairs) {
    // 1. Sketch Data
    CUDA_CHECK(cudaMalloc(&d_allSketches, totalSketchElements * sizeof(uint64_t)));

    // 2. Pair Indices
    CUDA_CHECK(cudaMalloc(&d_pairA_idx, numPairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pairB_idx, numPairs * sizeof(int)));

    // 3. Output Arrays
    CUDA_CHECK(cudaMalloc(&d_out_J, numPairs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_D, numPairs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_P, numPairs * sizeof(float)));
    
    // Zero out output arrays just to be safe
    CUDA_CHECK(cudaMemset(d_out_J, 0, numPairs * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_D, 0, numPairs * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_P, 0, numPairs * sizeof(float)));
}

/**
 * Transfers the flattened sketches and pairing indices from CPU to GPU.
 */
void GpuAligner::transferMashDataToDevice(const uint64_t* h_flatSketches, const int* h_pairA_idx, const int* h_pairB_idx, int totalSketchElements, int numPairs) {
    CUDA_CHECK(cudaMemcpy(d_allSketches, h_flatSketches, totalSketchElements * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pairA_idx, h_pairA_idx, numPairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pairB_idx, h_pairB_idx, numPairs * sizeof(int), cudaMemcpyHostToDevice));
}

/**
 * Retrieves the computed Jaccard, Distance, and P-value arrays back to the CPU.
 */
void GpuAligner::transferMashResultsToHost(float* h_out_J, float* h_out_D, float* h_out_P, int numPairs) {
    CUDA_CHECK(cudaMemcpy(h_out_J, d_out_J, numPairs * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_D, d_out_D, numPairs * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_P, d_out_P, numPairs * sizeof(float), cudaMemcpyDeviceToHost));
}

/**
 * Frees all device memory associated with the MASH calculations.
 */
void GpuAligner::freeMashMem() {
    CUDA_CHECK(cudaFree(d_allSketches));
    CUDA_CHECK(cudaFree(d_pairA_idx));
    CUDA_CHECK(cudaFree(d_pairB_idx));
    CUDA_CHECK(cudaFree(d_out_J));
    CUDA_CHECK(cudaFree(d_out_D));
    CUDA_CHECK(cudaFree(d_out_P));
}
