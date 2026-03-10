#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "newickHelper.hpp"

// Recursive helper to build the Newick tree string
std::string buildNewick(
    int node, 
    int numSeqs, 
    const std::vector<int>& left_child, 
    const std::vector<int>& right_child, 
    const std::vector<float>& dist_left, 
    const std::vector<float>& dist_right, 
    const std::vector<std::string>& seqNames) 
{
    // Base Case: If the node is a leaf (original sequence), return its name
    // Base Case: Check for valid sequence bounds
    if (node >= 0 && node < numSeqs) {
        return seqNames[node];
    } else if (node < 0) {
        // Return a fallback if the GPU left a dead-end branch
        fprintf(stdout, "\n --- In NewickHelper-----%d\n", node);
        return "NullNode"; 
    }
    
    // Recursive Step: It's an internal node
    int left = left_child[node];
    int right = right_child[node];
    
    // Clip negative branch lengths to 0.0 (standard for NJ artifacting)
    float d_left = std::max(0.0f, dist_left[node]);
    float d_right = std::max(0.0f, dist_right[node]);
    
    // Format floats to 6 decimal places cleanly
    std::ostringstream left_dist_ss, right_dist_ss;
    left_dist_ss << std::fixed << std::setprecision(6) << d_left;
    right_dist_ss << std::fixed << std::setprecision(6) << d_right;

    // Traverse left and right branches
    std::string left_str = buildNewick(left, numSeqs, left_child, right_child, dist_left, dist_right, seqNames) + ":" + left_dist_ss.str();
    std::string right_str = buildNewick(right, numSeqs, left_child, right_child, dist_left, dist_right, seqNames) + ":" + right_dist_ss.str();
    
    // Combine them wrapped in parentheses
    return "(" + left_str + "," + right_str + ")";
}