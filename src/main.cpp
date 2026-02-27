#include "masterHeader.cuh"
#include "twoBitCompressor.hpp"
#include "timer.hpp"
#include <iostream>
#include <cstdint>
#include <vector>
#include <boost/program_options.hpp>
#include "zlib.h"
#include <stdio.h>
#include "kseq.h"

// For parsing the command line values
namespace po = boost::program_options;

KSEQ_INIT(gzFile, gzread); //file pointer type for FASTA encoded file

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
    if ((kmerSize < 2) || (kmerSize > 50)) {
        std::cerr << "ERROR! kmerSize should be between 2 and 15." << std::endl;
        exit(1);
    }
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
    gzFile fp = gzopen("../dataset_dipper/t2.unaligned.fa", "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", refFilename.c_str());
        exit(1);
    }


    GpuAligner Aligner;

    
    kseq_t *seq = kseq_init(fp);
    int l;
    uint32_t seqCount = 0;

    while ((l = kseq_read(seq)) >= 0) {
        fprintf(stdout, "\n--- Sequence %u: %s (length: %zu) ---\n", seqCount, seq->name.s, seq->seq.l);

        uint32_t compressedSeqLen = (seq->seq.l + 15) / 16;
        uint32_t numKmers = (seq->seq.l >= kmerSize) ? (seq->seq.l - kmerSize + 1) : 0;

        fprintf(stdout, "KmerSize = %u, numKmers = %u\n", kmerSize, numKmers);

        std::vector<uint32_t> compressedSeq(compressedSeqLen);
        std::vector<size_t> kmerArr(numKmers);

        twoBitCompress(seq->seq.s, seq->seq.l, compressedSeq.data());

        fprintf(stdout, "Compressed sequence: ");
        for (uint32_t i = 0; i < compressedSeqLen; i++) {
            fprintf(stdout, "%08x ", compressedSeq[i]);
        }
        fprintf(stdout, "\n");

        Aligner.allocateMem(compressedSeqLen, numKmers, kmerSize);
        Aligner.seedTableOnGpu(compressedSeq.data(), compressedSeqLen, kmerSize, numKmers, kmerArr.data());

        fprintf(stdout, "Kmers: ");
        for (uint32_t i = 0; i < numKmers; i++) {
            fprintf(stdout, "%08lx ", kmerArr[i]);
        }
        fprintf(stdout, "\n");

        seqCount++;
    }

    if (seqCount == 0) {
        fprintf(stderr, "ERROR: No sequences found in file.\n");
        exit(1);
    }
    fprintf(stdout, "\nProcessed %u sequences total.\n", seqCount);
}

