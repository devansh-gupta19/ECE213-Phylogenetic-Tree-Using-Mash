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
    uint32_t bottomK;
    uint32_t maxReads;
    uint32_t batchSize;
    uint32_t numThreads;

    // Parse the command line options
    po::options_description desc{"Options"};
    desc.add_options()
    // Arguments for constructing seedTable
    ("kmerSize,k", po::value<uint32_t>(&kmerSize)->default_value(14), "Minimizer seed size (range: 2-15)")
    // Arguments for readMapper
    ("reads,q", po::value<std::string>(&readsFilename), "Input read sequences in FASTA file format.")
    ("readSize,s", po::value<uint32_t>(&readSize)->default_value(256), "Size of the read in number of bases [DO NOT CHANGE THE DEFAULT OF 256!]")
    ("bottomK,w", po::value<uint32_t>(&bottomK)->default_value(100), "Size of MASH sketch (range: 1-32)")
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
    if ((kmerSize < 2) || (kmerSize > 32)) {
        std::cerr << "ERROR! kmerSize should be between 2 and 32." << std::endl;
        exit(1);
    }
    // if ((bottomK < 1) || (bottomK > 32)) {
    //     std::cerr << "ERROR! bottomK should be between 1 and 64." << std::endl;
    //     exit(1);
    // }
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

    std::vector<std::vector<uint64_t>> allSketches;
    std::vector<std::string> seqNames;
    kseq_t *seq = kseq_init(fp);
    int l;
    uint32_t seqCount = 0;
    uint32_t totReads = 0;

    while ((l = kseq_read(seq)) >= 0 && (totReads < maxReads)) {
        totReads++;
        fprintf(stdout, "\n--- Sequence %u: %s (length: %zu) ---\n", seqCount, seq->name.s, seq->seq.l);

        uint32_t compressedSeqLen = (seq->seq.l + 15) / 16;
        uint32_t numKmers = (seq->seq.l >= kmerSize) ? (seq->seq.l - kmerSize + 1) : 0;

        fprintf(stdout, "KmerSize = %u, numKmers = %u\n", kmerSize, numKmers);

        std::vector<uint32_t> compressedSeq(compressedSeqLen);
        std::vector<size_t> kmerArr(numKmers);
    
        uint32_t actualSketchSize = (numKmers < bottomK) ? numKmers : bottomK;
        std::vector<uint64_t> hOut_sketch(actualSketchSize);

        twoBitCompress(seq->seq.s, seq->seq.l, compressedSeq.data());

        // fprintf(stdout, "Compressed sequence: ");
        // for (uint32_t i = 0; i < compressedSeqLen; i++) {
        //     fprintf(stdout, "%08x ", compressedSeq[i]);
        // }
        // fprintf(stdout, "\n");

        Aligner.allocateMem(compressedSeqLen, numKmers, kmerSize);
        uint32_t numUniqueKmers = Aligner.seedTableOnGpu(compressedSeq.data(), compressedSeqLen, kmerSize, numKmers, kmerArr.data());
        // fprintf(stdout, "Unique numKmers = %u\n", numUniqueKmers);
        // fprintf(stdout, "Kmers: ");
        // for (uint32_t i = 0; i < numKmers; i++) {
        //     fprintf(stdout, "%08lx ", kmerArr[i]);
        // }

        Aligner.MurmurHashCaller(numUniqueKmers, 8, bottomK, hOut_sketch.data());
        // fprintf(stdout, "MASH Sketch (Bottom %u hashes): \n", actualSketchSize);
        // for (uint32_t i = 0; i < actualSketchSize; i++) {
        //     fprintf(stdout, "%016lx ", hOut_sketch[i]);
        // }

        // Store the valid sketch and sequence name for later pairwise comparison
        allSketches.push_back(hOut_sketch);
        seqNames.push_back(std::string(seq->name.s));

        fprintf(stdout, "\n");
        Aligner.clearAndReset();
        seqCount++;
    } // End of while loop to calculate MASH sketches for all sequences

    if (seqCount == 0) {
        fprintf(stderr, "ERROR: No sequences found in file.\n");
        exit(1);
    }
    fprintf(stdout, "\nProcessed %u sequences total.\n", seqCount);

    fprintf(stdout, "\n\n\n");
    // Pairwise MASH Distance Computation
    int numSequences = allSketches.size();
    if (numSequences < 2) {
        fprintf(stderr, "Need at least 2 sequences to compare.\n");
        return 1;
    }

    int sketchSize = bottomK; // Assuming all valid sketches are size bottomK
    // Number of pairs = nC2
    int numPairs = (numSequences * (numSequences - 1)) / 2;

    // 1. Flatten all sketches into a single 1D vector
    std::vector<uint64_t> flatSketches(numSequences * sketchSize);
    for (int i = 0; i < numSequences; ++i) {
        std::copy(allSketches[i].begin(), allSketches[i].end(), flatSketches.begin() + (i * sketchSize));
    }

    // 2. Build the pair indexing arrays for the Upper Triangular Matrix
    std::vector<int> h_pairA_idx(numPairs);
    std::vector<int> h_pairB_idx(numPairs);
    
    int pairCount = 0;
    for (int i = 0; i < numSequences; ++i) {
        for (int j = i + 1; j < numSequences; ++j) {
            h_pairA_idx[pairCount] = i;
            h_pairB_idx[pairCount] = j;
            pairCount++;
        }
    }

    // 3. Allocate Host memory for the results
    std::vector<float> h_out_J(numPairs);
    std::vector<float> h_out_D(numPairs);
    std::vector<float> h_out_P(numPairs);

    fprintf(stdout, "Data preparation completed in %ld msec.\n", timer.Stop());
    fprintf(stdout, "Launching GPU kernel for %d pairs...\n", numPairs);
    timer.Start();

    int totalSketchElements = flatSketches.size();
    Aligner.allocateMashMem(totalSketchElements, numPairs);
    Aligner.transferMashDataToDevice(flatSketches.data(), h_pairA_idx.data(), h_pairB_idx.data(), totalSketchElements, numPairs);

    Aligner.MashDistanceCalculationCaller(numPairs, sketchSize, kmerSize);

    Aligner.transferMashResultsToHost(h_out_J.data(), h_out_D.data(), h_out_P.data(), numPairs);

    fprintf(stdout, "GPU Pairwise calculation completed in %ld msec.\n", timer.Stop());

    // 4. Print Results
    for (int p = 0; p < numPairs; ++p) {
        int idxA = h_pairA_idx[p];
        int idxB = h_pairB_idx[p];
        fprintf(stdout, "[%s] vs [%s] | Jaccard: %.4f | Distance: %.4f | P-Value: %.2e\n", 
                seqNames[idxA].c_str(), seqNames[idxB].c_str(), h_out_J[p], h_out_D[p], h_out_P[p]);
    }
    Aligner.freeMashMem();


    kseq_destroy(seq);
    gzclose(fp);
}

