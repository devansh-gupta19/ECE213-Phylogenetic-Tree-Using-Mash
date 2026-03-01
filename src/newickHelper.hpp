#ifndef NEWICK_HELPER_HPP
#define NEWICK_HELPER_HPP

#include <string>
#include <vector>

// Function declaration
std::string buildNewick(
    int node, 
    int numSeqs, 
    const std::vector<int>& left_child, 
    const std::vector<int>& right_child, 
    const std::vector<float>& dist_left, 
    const std::vector<float>& dist_right, 
    const std::vector<std::string>& seqNames
);

#endif // NEWICK_HELPER_HPP