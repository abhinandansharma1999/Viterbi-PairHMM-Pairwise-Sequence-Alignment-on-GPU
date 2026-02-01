#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

// structure to hold simple sequence data
struct FastaEntry {
    string header;
    string sequence;
};

// Function to remove gaps from a sequence
string removeGaps(const string& input) {
    string result;
    result.reserve(input.length()); // Optimize allocation
    for (char c : input) {
        if (c != '-') {
            result += c;
        }
    }
    return result;
}

// Robust FASTA reader (handles multiline sequences)
vector<FastaEntry> readFasta(const string& filepath) {
    vector<FastaEntry> entries;
    ifstream file(filepath);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filepath << endl;
        exit(1);
    }

    string line;
    string currentHeader;
    string currentSeq;
    bool parsingStarted = false;

    while (getline(file, line)) {
        // Handle Windows CR
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (line[0] == '>') {
            if (parsingStarted) {
                entries.push_back({currentHeader, currentSeq});
                currentSeq = "";
            }
            currentHeader = line.substr(1);
            parsingStarted = true;
        } else {
            currentSeq += line;
        }
    }
    // Add the final entry
    if (parsingStarted) {
        entries.push_back({currentHeader, currentSeq});
    }

    return entries;
}

int main(int argc, char* argv[]) {
    string rawFile, alignFile;

    // 1. Setup Program Options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("raw,r", po::value<string>(&rawFile)->required(), "Raw Sequences File (Source)")
        ("alignment,a", po::value<string>(&alignFile)->required(), "Alignment File (with gaps)")
        ("verbose,v", "Show all failed results")
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
    if (verbose) cout << "Reading Raw file: " << rawFile << "..." << endl;
    vector<FastaEntry> rawSeqs = readFasta(rawFile);
    
    if (verbose) cout << "Reading Alignment file: " << alignFile << "..." << endl;
    vector<FastaEntry> alignSeqs = readFasta(alignFile);

    // 3. Validation Logic
    if (rawSeqs.size() < alignSeqs.size()) {
        cerr << "[CRITICAL ERROR] The Alignment file has MORE sequences than the Raw file!" << endl;
        cerr << "Raw: " << rawSeqs.size() << ", Aligned: " << alignSeqs.size() << endl;
        return 1;
    }

    if (verbose) {
        cout << "Verifying " << alignSeqs.size() << " aligned sequences against the first " 
             << alignSeqs.size() << " raw sequences..." << endl;
        cout << string(60, '-') << endl;
    }

    int failures = 0;
    
    for (size_t i = 0; i < alignSeqs.size(); ++i) {
        string original = rawSeqs[i].sequence;
        string alignedWithGaps = alignSeqs[i].sequence;
        string restored = removeGaps(alignedWithGaps);

        if (original != restored) {
            failures++;
            if (verbose) {
                cout << "[FAIL] Sequence Index " << i + 1 << " (" << alignSeqs[i].header << ")" << endl;
                cout << "  Expected Length: " << original.length() << endl;
                cout << "  Restored Length: " << restored.length() << endl;
            }
            
            // Find first mismatch index
            size_t minLen = min(original.length(), restored.length());
            for(size_t k = 0; k < minLen; ++k) {
                if (original[k] != restored[k]) {
                    if (verbose) {
                        cout << "  First mismatch at char " << k << ": Expected '" 
                             << original[k] << "', Got '" << restored[k] << "'" << endl;
                    }
                    break;
                }
            }
            if (verbose) cout << string(60, '-') << endl;
        }
    }

    if (failures == 0) {
        cout << "[SUCCESS] All " << alignSeqs.size() << " aligned sequences match their raw originals.\n" << endl;
        return 0;
    } else {
        cout << "[FAILURE] Found " << failures << " inconsistencies." << endl;
        return 1;
    }
}