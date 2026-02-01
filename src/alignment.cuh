#include <stdint.h>
#include <vector>
#include <string>
#include <assert.h>
#include "timer.hpp"

void printGpuProperties();

using TB_PATH = std::vector<uint8_t>;

struct Sequence {
    int id;
    std::string name; 
    std::string seq;  // unaligned sequence
    std::string aln;  // aligned sequence
    Sequence(int _id, std::string _name, std::string _seq): id(_id), name(_name), seq(_seq) {};
};

struct GpuAligner {
    std::vector<Sequence> seqs;
    int32_t longestLen;
    int32_t numPairs;

    char*    d_seqs;
    uint8_t* d_tb;
    int32_t* d_seqLen;
    int32_t* d_info;

    void alignment();
    void allocateMem();
    void transferSequence2Device();
    std::vector<uint8_t> transferTB2Host();
    void getAlignedSequences(TB_PATH& tb_paths);
    void writeAlignment(std::string fileName, bool append);
    void clearAndReset();
};