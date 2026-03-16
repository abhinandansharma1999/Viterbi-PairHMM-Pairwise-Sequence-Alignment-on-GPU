#include <iostream>
#include <fstream>
#include <vector>
#include <boost/program_options.hpp>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include "kseq.h"
#include "zlib.h"
#include "alignment.cuh"

// For parsing the command line values
namespace po = boost::program_options;

// For reading in the FASTA file
KSEQ_INIT2(, gzFile, gzread)

int main(int argc, char** argv) {
    // Timer below helps with the performance profiling (see timer.hpp for more details)
    Timer timer;
    timer.Start();

    std::string sequenceFilename;
    std::string outputFilename;
    uint32_t maxPairs;
    uint32_t numThreads;
    uint32_t batchSize;

    // Parse the command line options
    po::options_description desc{"Options"};
    desc.add_options()
    ("sequence,i", po::value<std::string>(&sequenceFilename)->required(), "Input sequences in FASTA file format [REQUIRED].")
    ("output,o", po::value<std::string>(&outputFilename)->required(), "Output alignments in FASTA file format [REQUIRED]")
    ("numThreads,T", po::value<uint32_t>(&numThreads)->default_value(4), "Number of Threads (range: 1-8)")
    ("maxPairs,N", po::value<uint32_t>(&maxPairs)->default_value(5000), "Maximum number of sequence pairs to read from the input sequence file")
    ("batchSize,b", po::value<uint32_t>(&batchSize)->default_value(1000), "Number of pairs in a batch. Note: each batch consists of 2x batchSize sequences.")
    ("help,h", "Print help messages");

    po::options_description allOptions;
    allOptions.add(desc);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(allOptions).run(), vm);
        po::notify(vm);
    } catch(std::exception &e) {
        std::cerr << desc << std::endl;
        exit(1);
    }

    // Check input values
    if ((numThreads < 1) || (numThreads > 8)) {
        std::cerr << "ERROR! numThreads should be between 1 and 8." << std::endl;
        exit(1);
    }
    if (maxPairs < 1 || maxPairs > 5000) {
        std::cerr << "Error: maxPairs should be between 1 and 5000." << std::endl;
        exit(1);
    }

    // Check output file
    std::ofstream outFile (outputFilename);
    if (!outFile) {
        fprintf(stderr, "ERROR: cant open file: %s\n", outputFilename.c_str());
        exit(1);
    }

    // Print GPU information
    fprintf(stdout, "Setting CPU threads to %u and printing GPU device properties.\n", numThreads);
    tbb::global_control init(tbb::global_control::max_allowed_parallelism, numThreads);
    printGpuProperties();

    
    GpuAligner Aligner;
    int totSequenceCount = 0;
    int currSequenceCount = 0;
    // Read in the read sequences
    gzFile qp = gzopen(sequenceFilename.c_str(), "r");
    if (!qp) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", sequenceFilename.c_str());
        exit(1);
    }
    kseq_t *read = kseq_init(qp);
    while ((kseq_read(read) >= 0) && (totSequenceCount < (maxPairs << 1))) {
        Aligner.seqs.push_back(Sequence(currSequenceCount, read->name.s, std::string(read->seq.s, read->seq.l)));
        currSequenceCount++;
        totSequenceCount++;
        if (currSequenceCount == (batchSize << 1) || totSequenceCount == (maxPairs << 1)) {
            Aligner.numPairs = currSequenceCount / 2;
            Aligner.alignment();
            // Write the alignment to the output file
            bool append = totSequenceCount > (batchSize << 1);
            Aligner.writeAlignment(outputFilename, append);
            // Clear memory
            Aligner.clearAndReset();
            currSequenceCount = 0;
        }
    }
    fprintf(stdout, "\nProgram completed in %ld ms\n\n", timer.Stop());
    return 0;
}

