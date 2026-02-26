
void seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,

    uint32_t kmerSize,

    uint32_t* kmerOffset,
    size_t* kmerArr);

__global__ void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, uin32_t *hOut);