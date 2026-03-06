#include "alignment.h"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// CPU replacements for GPU types / intrinsics
// ─────────────────────────────────────────────────────────────────────────────
// Removed: #include <cuda_runtime.h> (implicit in .cu files)
// All cudaMalloc / cudaMemcpy / cudaFree calls are replaced with
// std::vector-based host allocations.  The d_* device pointers in the
// GpuAligner class are reinterpreted as plain host pointers of the same type.

// ─────────────────────────────────────────────────────────────────────────────
// allocateMem
// ─────────────────────────────────────────────────────────────────────────────
// CHANGE: cudaMalloc → new[] (plain heap allocation).
//         cudaGetErrorString / exit() error path removed; allocation failures
//         are reported via fprintf + exit as before, but using standard C++.
void GpuAligner::allocateMem() {
    longestLen = std::max_element(seqs.begin(), seqs.end(), [](const Sequence& a, const Sequence& b) {
        return a.seq.size() < b.seq.size();
    })->seq.size();

    // Flat sequence array
    d_seqs = new char[numPairs * 2 * longestLen]();
    if (!d_seqs) { fprintf(stderr, "CPU_ERROR: alloc d_seqs\n"); exit(1); }

    // Sequence lengths
    d_seqLen = new int32_t[numPairs * 2]();
    if (!d_seqLen) { fprintf(stderr, "CPU_ERROR: alloc d_seqLen\n"); exit(1); }

    // Traceback buffer
    int tb_length = longestLen << 1;
    d_tb = new uint8_t[numPairs * tb_length]();
    if (!d_tb) { fprintf(stderr, "CPU_ERROR: alloc d_tb\n"); exit(1); }

    // Meta-info [numPairs, longestLen]
    d_info = new int32_t[2]();
    if (!d_info) { fprintf(stderr, "CPU_ERROR: alloc d_info\n"); exit(1); }
}

// ─────────────────────────────────────────────────────────────────────────────
// transferSequence2Device
// ─────────────────────────────────────────────────────────────────────────────
// CHANGE: cudaMemcpy(HostToDevice) → std::memcpy into the same host arrays.
//         The "device" buffers are now just host memory, so the copy is local.
void GpuAligner::transferSequence2Device() {
    // Flatten sequences
    for (size_t i = 0; i < numPairs * 2; ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(d_seqs + (i * longestLen), s.data(), s.size());
    }

    // Sequence lengths
    for (int i = 0; i < (int)(numPairs * 2); ++i)
        d_seqLen[i] = (int32_t)seqs[i].seq.size();

    // Traceback buffer already zero-initialised by new[]() above

    // Meta info
    d_info[0] = numPairs;
    d_info[1] = longestLen;
}

// ─────────────────────────────────────────────────────────────────────────────
// transferTB2Host
// ─────────────────────────────────────────────────────────────────────────────
// CHANGE: cudaMemcpy(DeviceToHost) → std::memcpy from the same host buffer.
TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen << 1;
    TB_PATH h_tb(tb_length * numPairs);
    std::memcpy(h_tb.data(), d_tb, tb_length * numPairs * sizeof(uint8_t));
    return h_tb;
}

// ─────────────────────────────────────────────────────────────────────────────
// alignmentOnCPU  (replaces __global__ alignmentOnGPU)
// ─────────────────────────────────────────────────────────────────────────────
// CHANGES vs GPU kernel:
//  • __global__, __shared__, __syncthreads(), blockIdx, threadIdx removed.
//  • Formerly "shared" arrays become plain local std::vector / C arrays.
//  • The outer "if (bx==0 && tx==0)" guard is gone; function runs directly.
//  • min() / max() use std::min / std::max (no CUDA device versions needed).
//  • Inner wavefront loop runs sequentially (no thread-level parallelism).
static void alignmentOnCPU(
    int32_t* d_info,
    int32_t* d_seqLen,
    char*    d_seqs,
    uint8_t* d_tb)
{
    const int T = 10;
    const int O = 3;

    const int16_t MATCH    =  2;
    const int16_t MISMATCH = -1;
    const int16_t GAP      = -2;

    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    // Local buffers (were __shared__ in GPU version)
    std::vector<uint8_t> tbDir(T * T);
    std::vector<int16_t> wf_scores(3 * (T + 1));
    std::vector<uint8_t> localPath(2 * T);

    int32_t numPairs  = d_info[0];
    int32_t maxSeqLen = d_info[1];

    for (int pair = 0; pair < numPairs; ++pair) {

        bool    lastTile          = false;
        int16_t maxScore          = 0;
        int32_t currentPairPathLen = 0;
        int32_t reference_idx     = 0;
        int32_t query_idx         = 0;
        int32_t best_ti, best_tj;

        int32_t refStart        = (pair * 2)     * maxSeqLen;
        int32_t qryStart        = (pair * 2 + 1) * maxSeqLen;
        int32_t tbGlobalOffset  = pair * (maxSeqLen * 2);

        int32_t refTotalLen = d_seqLen[2 * pair];
        int32_t qryTotalLen = d_seqLen[2 * pair + 1];

        while (!lastTile) {
            int32_t refLen = std::min(T, (int)(refTotalLen - reference_idx));
            int32_t qryLen = std::min(T, (int)(qryTotalLen - query_idx));

            if ((reference_idx + refLen == refTotalLen) &&
                (query_idx     + qryLen == qryTotalLen))
                lastTile = true;

            // Reset wavefront scores
            std::fill(wf_scores.begin(), wf_scores.end(), (int16_t)-9999);

            best_ti = refLen;
            best_tj = qryLen;

            // Wavefront (diagonal) scoring loop
            for (int k = 0; k <= refLen + qryLen; ++k) {
                int curr_k   = (k % 3)       * (T + 1);
                int pre_k    = ((k + 2) % 3) * (T + 1);
                int prepre_k = ((k + 1) % 3) * (T + 1);

                int i_start = std::max(0, k - (int)qryLen);
                int i_end   = std::min((int)refLen, k);

                for (int i = i_start; i <= i_end; ++i) {
                    int j = k - i;

                    int16_t score     = -9999;
                    uint8_t direction = DIR_DIAG;

                    if (i == 0 && j == 0) {
                        score    = maxScore;
                        maxScore = -9999;
                    } else if (i == 0) {
                        score     = wf_scores[pre_k + i] + GAP;
                        direction = DIR_LEFT;
                    } else if (j == 0) {
                        score     = wf_scores[pre_k + (i - 1)] + GAP;
                        direction = DIR_UP;
                    } else {
                        char r_char = d_seqs[refStart + reference_idx + (i - 1)];
                        char q_char = d_seqs[qryStart + query_idx     + (j - 1)];

                        int16_t score_diag = wf_scores[prepre_k + (i - 1)] + (r_char == q_char ? MATCH : MISMATCH);
                        int16_t score_up   = wf_scores[pre_k    + (i - 1)] + GAP;
                        int16_t score_left = wf_scores[pre_k    + i]       + GAP;

                        score     = score_diag;
                        direction = DIR_DIAG;
                        if (score_up   > score) { score = score_up;   direction = DIR_UP;   }
                        if (score_left > score) { score = score_left; direction = DIR_LEFT; }
                    }

                    wf_scores[curr_k + i] = score;
                    if (i > 0 && j > 0)
                        tbDir[(i - 1) * T + (j - 1)] = direction;

                    // GACT overlap tracking
                    if (!lastTile) {
                        if (i > (refLen - O) && j > (qryLen - O)) {
                            if (score >= maxScore) {
                                maxScore = score;
                                best_ti  = i;
                                best_tj  = j;
                            }
                        }
                    }
                }
            } // end wavefront loop

            // Traceback
            int ti = (!lastTile) ? best_ti : refLen;
            int tj = (!lastTile) ? best_tj : qryLen;

            int next_ref_advance = ti;
            int next_qry_advance = tj;

            int localLen = 0;
            while (ti > 0 || tj > 0) {
                uint8_t dir;
                if      (ti == 0) { dir = DIR_LEFT; }
                else if (tj == 0) { dir = DIR_UP;   }
                else              { dir = tbDir[(ti - 1) * T + (tj - 1)]; }

                localPath[localLen++] = dir;

                if      (dir == DIR_DIAG) { ti--; tj--; }
                else if (dir == DIR_UP)   { ti--;        }
                else                      {       tj--;  }
            }

            // Write forward to output buffer
            for (int k = localLen - 1; k >= 0; --k) {
                d_tb[tbGlobalOffset + currentPairPathLen] = localPath[k];
                currentPairPathLen++;
            }

            reference_idx += next_ref_advance;
            query_idx     += next_qry_advance;
        } // end tile loop
    } // end pair loop
}

// ─────────────────────────────────────────────────────────────────────────────
// getAlignedSequences  — unchanged logic
// ─────────────────────────────────────────────────────────────────────────────
void GpuAligner::getAlignedSequences(TB_PATH& tb_paths) {
    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    int tb_length = longestLen << 1;

    for (int pair = 0; pair < numPairs; ++pair) {
        int tb_start = tb_length * pair;

        int seqId0 = 2 * pair;
        int seqId1 = 2 * pair + 1;
        std::string seq0 = seqs[seqId0].seq;
        std::string seq1 = seqs[seqId1].seq;
        std::string aln0 = "";
        std::string aln1 = "";
        int seqPos0 = 0;
        int seqPos1 = 0;

        for (int i = tb_start; i < tb_start + tb_length; ++i) {
            if (tb_paths[i] == DIR_DIAG) {
                aln0 += seq0[seqPos0]; aln1 += seq1[seqPos1];
                seqPos0++; seqPos1++;
            } else if (tb_paths[i] == DIR_UP) {
                aln0 += seq0[seqPos0]; aln1 += '-';
                seqPos0++;
            } else if (tb_paths[i] == DIR_LEFT) {
                aln0 += '-'; aln1 += seq1[seqPos1];
                seqPos1++;
            } else {
                break;
            }
        }

        seqs[seqId0].aln = aln0;
        seqs[seqId1].aln = aln1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// clearAndReset
// ─────────────────────────────────────────────────────────────────────────────
// CHANGE: cudaFree → delete[]
void GpuAligner::clearAndReset() {
    delete[] d_seqs;
    delete[] d_seqLen;
    delete[] d_tb;
    delete[] d_info;
    seqs.clear();
    longestLen = 0;
    numPairs   = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// alignment  (main orchestration)
// ─────────────────────────────────────────────────────────────────────────────
// CHANGE: kernel launch <<<numBlocks, blockSize>>> replaced by a direct
//         function call to alignmentOnCPU().  cudaGetLastError and
//         cudaDeviceSynchronize removed.
void GpuAligner::alignment() {
    allocateMem();
    transferSequence2Device();

    // Direct CPU call — replaces: alignmentOnGPU<<<1,1>>>(...)
    alignmentOnCPU(d_info, d_seqLen, d_seqs, d_tb);

    TB_PATH tb_paths = transferTB2Host();
    getAlignedSequences(tb_paths);
}

// ─────────────────────────────────────────────────────────────────────────────
// writeAlignment  — unchanged
// ─────────────────────────────────────────────────────────────────────────────
void GpuAligner::writeAlignment(std::string fileName, bool append) {
    std::ofstream outFile;
    if (append) outFile.open(fileName, std::ios::app);
    else        outFile.open(fileName);
    if (!outFile) {
        fprintf(stderr, "ERROR: cant open file: %s\n", fileName.c_str());
        exit(1);
    }
    for (auto& seq : seqs) {
        outFile << ('>' + seq.name + '\n');
        outFile << (seq.aln + '\n');
    }
    outFile.close();
}