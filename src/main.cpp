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
#include <string>
#include "newickHelper.hpp"

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
    ("bottomK,w", po::value<uint32_t>(&bottomK)->default_value(100), "Number of smallest hashed Kmers retained in the sketch (range: 10-9000)")
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
    if ((numThreads < 1) || (numThreads > 8)) {
        std::cerr << "ERROR! numThreads should be between 1 and 8." << std::endl;
        exit(1);
    }
    if ((bottomK < 10) || (bottomK > 9000)) {
        std::cerr << "ERROR! bottomK should be between 1 and 9000." << std::endl;
        exit(1);
    }
    if (batchSize == 0) {
        std::cerr << "ERROR! batchSize must be greater than 0." << std::endl;
        exit(1);
    }

    // Print GPU information
    fprintf(stdout, "Setting CPU threads to %u and printing GPU device properties.\n", numThreads);
    printGpuProperties();

    // Read reference sequence as kseq_t object
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
    uint32_t batchCount = 0;

    bool endOfFile = false;

    // Batch processing loop
    while (!endOfFile && totReads < maxReads) {
        std::vector<std::string> batchSeqs;
        std::vector<std::string> batchNames;
        
        // 1. Read up to 'batchSize' sequences into CPU memory
        for (uint32_t b = 0; b < batchSize && totReads < maxReads; ++b) {
            l = kseq_read(seq);
            if (l >= 0) {
                batchSeqs.push_back(std::string(seq->seq.s, seq->seq.l));
                batchNames.push_back(std::string(seq->name.s));
                totReads++;
            } else {
                endOfFile = true;
                break;
            }
        }

        // If the batch is empty, we are done
        if (batchSeqs.empty()) {
            break;
        }

        batchCount++;
        fprintf(stdout, "\n=======================================================\n");
        fprintf(stdout, "Processing Batch %u: %zu sequences (Total read so far: %u)\n", batchCount, batchSeqs.size(), totReads);
        fprintf(stdout, "=======================================================\n");

        // 2. Process the loaded batch sequentially on the GPU
        for (size_t i = 0; i < batchSeqs.size(); ++i) {
            size_t currentSeqLen = batchSeqs[i].length();
            
            fprintf(stdout, "\n--- Sequence %u: %s (length: %zu) ---\n", seqCount, batchNames[i].c_str(), currentSeqLen);

            uint32_t compressedSeqLen = (currentSeqLen + 15) / 16;
            uint32_t numKmers = (currentSeqLen >= kmerSize) ? (currentSeqLen - kmerSize + 1) : 0;

            fprintf(stdout, "KmerSize = %u, numKmers = %u\n", kmerSize, numKmers);

            std::vector<uint32_t> compressedSeq(compressedSeqLen);
            std::vector<size_t> kmerArr(numKmers);
        
            uint32_t actualSketchSize = (numKmers < bottomK) ? numKmers : bottomK;
            std::vector<uint64_t> hOut_sketch(actualSketchSize);

            // Compress the specific sequence from our batch array
            twoBitCompress(const_cast<char*>(batchSeqs[i].c_str()), currentSeqLen, compressedSeq.data());
            Aligner.allocateMem(compressedSeqLen, numKmers, kmerSize);
            uint32_t numUniqueKmers = Aligner.seedTableOnGpu(compressedSeq.data(), compressedSeqLen, kmerSize, numKmers, kmerArr.data());

            // MinHash the Kmers and retain only bottomk
            Aligner.MurmurHashCaller(numUniqueKmers, 8, bottomK, hOut_sketch.data());

            // Store the valid sketch and sequence name for later pairwise comparison
            allSketches.push_back(hOut_sketch);
            seqNames.push_back(batchNames[i]);

            Aligner.clearAndReset();
            seqCount++;
        }
    } // End of batch loop

    if (seqCount == 0) {
        fprintf(stderr, "ERROR: No sequences found in file.\n");
        exit(1);
    }
    fprintf(stdout, "\nProcessed %u sequences total across %u batches.\n", seqCount, batchCount);

    fprintf(stdout, "\n\n");
    
    // Pairwise MASH Distance Computation
    int numSequences = allSketches.size();
    if (numSequences < 2) {
        fprintf(stderr, "Need at least 2 sequences to compare.\n");
        return 1;
    }

    // Assuming all valid sketches are size bottomK
    int sketchSize = bottomK;
    // Number of pairs = nC2
    int numPairs = (numSequences * (numSequences - 1)) / 2;

    // Flatten all sketches into a single 1D vector
    std::vector<uint64_t> flatSketches(numSequences * sketchSize);
    for (int i = 0; i < numSequences; ++i) {
        std::copy(allSketches[i].begin(), allSketches[i].end(), flatSketches.begin() + (i * sketchSize));
    }

    // Build the pair indexing arrays for the Upper Triangular Matrix
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

    // Allocate host memory for the results of tree topology
    int totalNodes = 2 * numSequences - 1;
    std::vector<int> h_left_child(totalNodes);
    std::vector<int> h_right_child(totalNodes);
    std::vector<float> h_dist_left(totalNodes);
    std::vector<float> h_dist_right(totalNodes);

    fprintf(stdout, "Launching GPU kernel for %d pairs...\n", numPairs);
    timer.Start();

    int totalSketchElements = flatSketches.size();
    Aligner.allocateMashMem(totalSketchElements, numPairs);
    Aligner.transferMashDataToDevice(flatSketches.data(), h_pairA_idx.data(), h_pairB_idx.data(), totalSketchElements, numPairs);

    // Call the updated pipeline (Mash -> NJ)
    Aligner.MashDistanceCalculationCaller(
        numPairs, sketchSize, kmerSize, numSequences,
        h_left_child.data(), h_right_child.data(), h_dist_left.data(), h_dist_right.data()
    );

    fprintf(stdout, "GPU Pairwise & NJ calculation completed in %ld msec.\n", timer.Stop());
    
    // Print Neighbor-Joining Tree Results
    fprintf(stdout, "\n--- Neighbor-Joining Tree Topology ---\n");
    for (int i = numSequences; i < totalNodes - 1; ++i) {
        // Resolve names: If the child index is < numSequences, it's a leaf node, so print its actual FASTA name. 
        // Otherwise, it's an internal node, so print "Node X".
        std::string leftName = (h_left_child[i] < numSequences) ? seqNames[h_left_child[i]] : "Node " + std::to_string(h_left_child[i]);
        std::string rightName = (h_right_child[i] < numSequences) ? seqNames[h_right_child[i]] : "Node " + std::to_string(h_right_child[i]);

        fprintf(stdout, "Internal Node %d joined:\n", i);
        fprintf(stdout, "  -> %s (Branch Length: %.4f)\n", leftName.c_str(), h_dist_left[i]);
        fprintf(stdout, "  -> %s (Branch Length: %.4f)\n", rightName.c_str(), h_dist_right[i]);
    }
    
    fprintf(stdout, "\n --- Tree creation complete-----\n");

    //Newick tree creation
    int maxPopulatedNode = (2 * numSequences) - 3;

    // 1. Find the two nodes that were never assigned as children (the final two subtrees)
    std::vector<bool> is_child(maxPopulatedNode + 1, false);
    for (int i = numSequences; i <= maxPopulatedNode; ++i) {
        if (h_left_child[i] >= 0) is_child[h_left_child[i]] = true;
        if (h_right_child[i] >= 0) is_child[h_right_child[i]] = true;
    }

    std::vector<int> final_roots;
    for (int i = 0; i <= maxPopulatedNode; ++i) {
        if (!is_child[i]) {
            final_roots.push_back(i);
        }
    }

    // 2. Build the final string by joining the remaining subtrees
    std::string newickTree;
    if (final_roots.size() == 2) {
        std::string tree1 = buildNewick(final_roots[0], numSequences, h_left_child, h_right_child, h_dist_left, h_dist_right, seqNames);
        std::string tree2 = buildNewick(final_roots[1], numSequences, h_left_child, h_right_child, h_dist_left, h_dist_right, seqNames);
        
        // Wrap them together to form the unrooted Newick tree
        newickTree = "(" + tree1 + "," + tree2 + ");";
    } else {
        // Fallback just in case the GPU logic behaved unexpectedly
        fprintf(stderr, "\nWARNING: Expected 2 final subtrees, found %zu. Falling back to max populated node.\n", final_roots.size());
        newickTree = buildNewick(maxPopulatedNode, numSequences, h_left_child, h_right_child, h_dist_left, h_dist_right, seqNames) + ";";
    }

    // Save to file
    std::ofstream nwkFile("phylogenetic_tree.nwk");
    if (nwkFile.is_open()) {
        nwkFile << newickTree << "\n";
        nwkFile.close();
        fprintf(stdout, "\nSuccessfully saved Newick tree to 'phylogenetic_tree.nwk'\n");
    } else {
        fprintf(stderr, "\nERROR: Could not open file to save Newick tree.\n");
    }
    //End of newick tree creation

    Aligner.freeMashMem();

    kseq_destroy(seq);
    gzclose(fp);
    return 0;
}