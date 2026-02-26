#include "masterHeader.cuh"
#include "twoBitCompressor.hpp"
#include "timer.hpp"
#include <iostream>
#include <cstdint>
#include <vector>
#include <boost/program_options.hpp>
#include "zlib.h"
#include <stdio.h>

// For parsing the command line values
namespace po = boost::program_options;

int main(int argc, char** argv) {
    // Timer below helps with the performance profiling (see timer.hpp for more details)
    Timer timer;

    std::string refFilename;
    std::string readsFilename;
    uint32_t readSize;
    uint32_t kmerSize;
    uint32_t kmerWindow;
    uint32_t maxReads;
    uint32_t batchSize;
    uint32_t numThreads;

    // Parse the command line options
    po::options_description desc{"Options"};
    desc.add_options()
    // Arguments for constructing seedTable
    ("reference,r", po::value<std::string>(&refFilename)->required(), "Input reference sequence in FASTA file format [REQUIRED].")
    ("kmerSize,k", po::value<uint32_t>(&kmerSize)->default_value(14), "Minimizer seed size (range: 2-15)")
    // Arguments for readMapper
    ("reads,q", po::value<std::string>(&readsFilename), "Input read sequences in FASTA file format.")
    ("readSize,s", po::value<uint32_t>(&readSize)->default_value(256), "Size of the read in number of bases [DO NOT CHANGE THE DEFAULT OF 256!]")
    ("kmerWindow,w", po::value<uint32_t>(&kmerWindow)->default_value(16), "Minimizer window size (range: 1-32)")
    ("maxReads,N", po::value<uint32_t>(&maxReads)->default_value(1e6), "Maximum number of reads to read from the input read sequence file")
    ("batchSize,b", po::value<uint32_t>(&batchSize)->default_value(32), "Number of reads in a batch")
    // Other options
    ("numThreads,T", po::value<uint32_t>(&numThreads)->default_value(4), "Number of Threads (range: 1-8)")
    ("help,h", "Print help messages");

    po::options_description allOptions;
    allOptions.add(desc);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(allOptions).run(), vm);
        po::notify(vm);
    } catch(std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << desc << std::endl;
        exit(1);
    }

    // Check input values
    // if ((kmerSize < 2) || (kmerSize > 50)) {
    //     std::cerr << "ERROR! kmerSize should be between 2 and 15." << std::endl;
    //     exit(1);
    // }
    if ((kmerWindow < 1) || (kmerWindow > 32)) {
        std::cerr << "ERROR! kmerWindow should be between 1 and 64." << std::endl;
        exit(1);
    }
    if ((numThreads < 1) || (numThreads > 8)) {
        std::cerr << "ERROR! numThreads should be between 1 and 8." << std::endl;
        exit(1);
    }

    // Print GPU information
    timer.Start();
    fprintf(stdout, "Setting CPU threads to %u and printing GPU device properties.\n", numThreads);
    // tbb::global_control init(tbb::global_control::max_allowed_parallelism, numThreads);
    printGpuProperties();
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    // Read reference sequence as kseq_t object
    timer.Start();
    fprintf(stdout, "Reading reference sequence and compressing to two-bit encoding.\n");
    gzFile fp = gzopen(refFilename.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", refFilename.c_str());
        exit(1);
    }


    GpuAligner Aligner;

    // Example usage: Hashing a 21-mer canonical string
    const char* seq = "AACGTCGATCGATCGATCGATCCGTACGTCGATCGATCGATCGATCCGATCGATCGAACGTCGATCGATCGAACGTCGATCGATCGATCGATCTCGATCTCACGTCGATCGATCGATCGATCGATC";
    uint32_t compressedSeq[(strlen(seq)+15)/16];

    uint32_t compressedSeqLen = (strlen(seq) + 15) / 16;
    uint32_t numKmers = strlen(seq) - kmerSize + 1;

    fprintf(stdout, "KmerSize = %d\n", kmerSize);
    fprintf(stdout, "compressedSeqLen = %d\n", compressedSeqLen);
    fprintf(stdout, "numKmers = %d\n", numKmers);
    size_t kmerArr[numKmers];

    twoBitCompress((char*)seq, strlen(seq), compressedSeq);

    fprintf(stdout, "Compressed sequence: ");

    for (uint32_t i = 0; i < compressedSeqLen; i++) {
        fprintf(stdout, "%08x ", compressedSeq[i]);
    }
    fprintf(stdout, "\n");

    Aligner.seedTableOnGpu (
        compressedSeq,
        compressedSeqLen,
        kmerSize,
        kmerArr);

    for (uint32_t i = 0; i < numKmers; i++) {
        fprintf(stdout, "%08x ", kmerArr[i]);
    }
    fprintf(stdout, "\n");


    return 0;
}

// int main() {
//     // Example usage: Hashing a 21-mer canonical string
//     const char* kmer = "ACGTCGATCGATCGATCGATC";

//     // MASH typically uses a constant seed like 42
//     uint32_t seed = 42; 

//     // Perform the hash
//     uint32_t hash_value = MurmurHash3_x86_32(kmer, strlen(kmer), seed);

//     std::cout << "Canonical k-mer: " << kmer << std::endl;
//     std::cout << "MurmurHash3 (32-bit) value: " << hash_value << std::endl;
    
//     return 0;
// }
