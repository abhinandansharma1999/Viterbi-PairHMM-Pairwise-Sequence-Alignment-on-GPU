#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

// Configuration for scoring
const int MATCH_SCORE = 2;
const int MISMATCH_SCORE = -1;
const int GAP_SCORE = -2;

struct AlignmentPair {
    string seqA;
    string seqB;
    string name; // Optional, for reporting
};

// Function to calculate alignment score
long long calculateScore(const string& a, const string& b) {
    long long score = 0;
    size_t len = min(a.length(), b.length());

    for (size_t i = 0; i < len; ++i) {
        char c1 = a[i];
        char c2 = b[i];

        if (c1 == '-' || c2 == '-') {
            score += GAP_SCORE;
        } else if (c1 == c2) {
            score += MATCH_SCORE;
        } else {
            score += MISMATCH_SCORE;
        }
    }
    return score;
}

// Helper to read FASTA or line-based files into pairs
vector<AlignmentPair> readAlignments(const string& filepath) {
    vector<AlignmentPair> alignments;
    ifstream file(filepath);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filepath << endl;
        exit(1);
    }

    // Temporary struct to hold parsed FASTA records
    struct FastaRecord {
        string header;
        string sequence;
    };
    
    vector<FastaRecord> records;
    string line;
    string currentHeader;
    string currentSeq;
    bool parsingStarted = false;

    // 1. Read the entire file and parse into FastaRecords
    while (getline(file, line)) {
        // Remove carriage returns if on Windows/mixed files
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (line[0] == '>') {
            // Save previous record if it exists
            if (parsingStarted) {
                records.push_back({currentHeader, currentSeq});
                currentSeq = ""; // Reset for next
            }
            // Start new record
            currentHeader = line.substr(1); // Remove '>'
            parsingStarted = true;
        } else {
            // It's a sequence line (append to handle multiline FASTA)
            // Optional: trim whitespace
            currentSeq += line;
        }
    }
    // Push the very last record found
    if (parsingStarted) {
        records.push_back({currentHeader, currentSeq});
    }

    // 2. Group records into pairs (1&2, 3&4, etc.)
    for (size_t i = 0; i < records.size(); i += 2) {
        if (i + 1 < records.size()) {
            // We have a pair (i and i+1)
            alignments.push_back({
                records[i].sequence,     // Seq 1
                records[i+1].sequence,   // Seq 2
                records[i].header        // Use the first header as the Pair Name
            });
        } else {
            cerr << "Warning: File " << filepath << " has an odd number of sequences. "
                 << "Ignoring the last orphan sequence: " << records[i].header << endl;
        }
    }

    return alignments;
}

int main(int argc, char* argv[]) {
    string refFile, estFile;

    // 1. Setup Program Options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("reference,r", po::value<string>(&refFile)->required(), "Reference alignment file")
        ("estimate,e", po::value<string>(&estFile)->required(), "Estimate alignment file (To Evaluate)")
        ("verbose,v", "Show all pair-wise comparisons")
        ("help,h", "Produce help message");

    po::variables_map vm;

    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        
        if (vm.count("help")) {
            cout << desc << "\n";
            return 0;
        }

        po::notify(vm);
    } catch (const po::error& ex) {
        if(argc == 1) {
            std::cerr << desc << std::endl;
            return 0;
        }
        cerr << "Error: " << ex.what() << endl;
        return 1;
    }

    bool verbose = vm.count("verbose");

    // 2. Read Files
    if (verbose) cout << "Reading Estimate file: " << estFile << "..." << endl;
    vector<AlignmentPair> estAligns = readAlignments(estFile);
    
    if (verbose) cout << "Reading Reference file: " << refFile << "..." << endl;
    vector<AlignmentPair> refAligns = readAlignments(refFile);

    // 3. Process Comparisons
    size_t count = min(estAligns.size(), refAligns.size());
    if (count == 0) {
        cerr << "Error: One of the files contains no valid alignment pairs." << endl;
        return 1;
    }

    double totalPercentageLoss = 0.0;
    int validComparisons = 0;

    if (verbose) {
        cout << "\n" << string(70, '-') << endl;
        cout << left << setw(10) << "ID" 
             << setw(15) << "Ref Score" 
             << setw(15) << "Est Score" 
             << setw(15) << "Diff" 
             << setw(15) << "% Loss" << endl;
        cout << string(70, '-') << endl;
    }

    for (size_t i = 0; i < count; ++i) {
        AlignmentPair& est = estAligns[i];
        AlignmentPair& ref = refAligns[i];

        // LOGIC: Truncate Reference to Estimate Length
        size_t estLen = est.seqA.length(); // Assuming seqA and seqB are same length in valid alignment
        
        // Truncate ref strings if they are longer than estimate
        string refA_trunc = ref.seqA.substr(0, min(ref.seqA.length(), estLen));
        string refB_trunc = ref.seqB.substr(0, min(ref.seqB.length(), estLen));

        // Calculate Scores
        long long scoreEst = calculateScore(est.seqA, est.seqB);
        long long scoreRef = calculateScore(refA_trunc, refB_trunc);

        long long diff = scoreRef - scoreEst;
        
        // Calculate Percentage Difference
        // Formula: (Ref - Est) / Ref * 100
        double percentLoss = 0.0;
        
        if (scoreRef != 0) {
            percentLoss = (double)diff / (double)abs(scoreRef) * 100.0;
        } else {
            // Edge case: Reference score is 0. 
            // If diff is 0, loss is 0. If diff is huge, loss is undefined/infinite.
            percentLoss = (diff == 0) ? 0.0 : 0.0; // Keeping 0 for safety in stats
        }

        totalPercentageLoss += percentLoss;
        validComparisons++;

        if (verbose) {
            cout << left << setw(10) << i+1 
                 << setw(15) << scoreRef 
                 << setw(15) << scoreEst 
                 << setw(15) << diff 
                 << fixed << setprecision(2) << percentLoss << "%" << endl;
        }
    }

    if (verbose) cout << string(70, '-') << endl;

    // 4. Final Average
    if (validComparisons > 0) {
        double avgLoss = totalPercentageLoss / validComparisons;
        cout << "Valid Comparisons: " << validComparisons << " pairs. ";
        cout << "Average Percentage Score Loss: " << avgLoss << "%\n" << endl;
    } else {
        cout << "No valid comparisons made." << endl;
    }

    return 0;
}