#include "alignment.h"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include <limits>

// =============================================================================
// HMM PARAMETERS  (all probabilities stored in log-space)
// =============================================================================
//
// PURPOSE OF THIS FILE
// --------------------
// This is the GOLD STANDARD reference implementation. It solves the FULL O(mn)
// Viterbi DP for each sequence pair — no GACT tiling, no approximation.
// Output is used to validate all GACT-based implementations by comparing
// alignment quality scores. Expected runtime: ~60 min for 5000 pairs on CPU.
//
// FULL DP vs GACT:
//   Full DP: allocates an (R+1)×(Q+1) matrix per pair — O(mn) memory per pair.
//            Guarantees globally optimal alignment under the HMM model.
//   GACT:    processes a chain of T×T tiles — O(T²) memory regardless of length.
//            Introduces a small accuracy loss (~0.05% on the test dataset).
//
// HMM STATE MODEL
// ---------------
//   M (0) – Match / Mismatch  → both sequences advance      → DIR_DIAG
//   I (1) – Insert in query   → query advances, ref gaps    → DIR_LEFT
//   D (2) – Delete from query → ref advances, query gaps    → DIR_UP
//
// Emission probabilities (state M only; I and D are "silent" for one sequence):
//   p(match)    = 0.90
//   p(mismatch) = 0.10
//
// Transition matrix (row = from-state, col = to-state {M, I, D}):
//   M → { M:0.90,  I:0.05,  D:0.05 }
//   I → { M:0.70,  I:0.30,  D:---  }   I→D forbidden (prevents alignment ambiguity)
//   D → { M:0.70,  I:---,   D:0.30 }   D→I forbidden
//
// Initial state probabilities:
//   π(M)=0.90, π(I)=0.05, π(D)=0.05
//
// All probabilities stored as natural logarithms (nats) to avoid underflow.
// =============================================================================

namespace HMM {
    // log(0): sentinel for unreachable cells and forbidden transitions
    static const double NEG_INF = -1e18;

    // Log-emission probabilities for state M
    static const double LOG_EMIT_MATCH    = std::log(0.90);
    static const double LOG_EMIT_MISMATCH = std::log(0.10);

    //  LOG_TRANS[from][to] — NEG_INF entries are forbidden transitions
    static const double LOG_TRANS[3][3] = {
        { std::log(0.90), std::log(0.05), std::log(0.05) }, // from M: M→M, M→I, M→D
        { std::log(0.70), std::log(0.30), NEG_INF         }, // from I: I→M, I→I, I→D(forbidden)
        { std::log(0.70), NEG_INF,        std::log(0.30)  }  // from D: D→M, D→I(forbidden), D→D
    };

    // LOG_INIT[s] — log probability of starting in state s at position (0,0)
    static const double LOG_INIT[3] = {
        std::log(0.90),  // M — most likely starting state
        std::log(0.05),  // I
        std::log(0.05)   // D
    };

    static const int S_M = 0;
    static const int S_I = 1;
    static const int S_D = 2;
}

// =============================================================================
// allocateMem
//
// Allocates flat CPU arrays using the same layout as the GPU versions so
// the alignment logic is portable. No tiling scratch is needed here since
// the full DP keeps the entire matrix in memory.
//
// d_tb worst-case size: 2*longestLen per pair (all gaps, no matches).
// =============================================================================
void GpuAligner::allocateMem() {
    longestLen = std::max_element(seqs.begin(), seqs.end(), [](const Sequence& a, const Sequence& b) {
        return a.seq.size() < b.seq.size();
    })->seq.size();

    d_seqs   = new char   [numPairs * 2 * longestLen]();
    if (!d_seqs)   { fprintf(stderr, "CPU_ERROR: alloc d_seqs\n");   exit(1); }

    d_seqLen = new int32_t[numPairs * 2]();
    if (!d_seqLen) { fprintf(stderr, "CPU_ERROR: alloc d_seqLen\n"); exit(1); }

    // Traceback buffer: worst case each cell emits one step → refLen + qryLen steps max
    int tb_length = longestLen << 1;
    d_tb     = new uint8_t [numPairs * tb_length]();
    if (!d_tb)     { fprintf(stderr, "CPU_ERROR: alloc d_tb\n");     exit(1); }

    d_info   = new int32_t[2]();
    if (!d_info)   { fprintf(stderr, "CPU_ERROR: alloc d_info\n");   exit(1); }
}

// =============================================================================
// transferSequence2Device  (host→host, no GPU)
//
// Named for API compatibility with GPU versions. Packs sequence data into
// the flat strided array layout expected by alignmentOnCPU.
// =============================================================================
void GpuAligner::transferSequence2Device() {
    for (size_t i = 0; i < (size_t)(numPairs * 2); ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(d_seqs + (i * longestLen), s.data(), s.size());
    }
    for (int i = 0; i < (int)(numPairs * 2); ++i)
        d_seqLen[i] = (int32_t)seqs[i].seq.size();

    d_info[0] = numPairs;
    d_info[1] = longestLen;
}

// =============================================================================
// transferTB2Host
//
// Named for API compatibility. Plain memcpy on CPU — no device transfer.
// =============================================================================
TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen << 1;
    TB_PATH h_tb(tb_length * numPairs);
    std::memcpy(h_tb.data(), d_tb, tb_length * numPairs * sizeof(uint8_t));
    return h_tb;
}

// =============================================================================
// alignmentOnCPU  —  Full DP matrix Viterbi pair-HMM  (no tiling)
// =============================================================================
//
// Computes the globally optimal Viterbi alignment for each pair by filling
// the complete (R+1)×(Q+1) DP matrix. This is the reference used to measure
// the accuracy loss of all GACT-tiled implementations.
//
// VITERBI RECURRENCES (1-based i over ref, 1-based j over query):
//
//   V_M[i][j] = logEmit(r[i], q[j])
//               + max_s( V_s[i-1][j-1] + logT[s→M] )   ← diagonal predecessor
//
//   V_I[i][j] = max( V_M[i][j-1] + logT[M→I],
//                    V_I[i][j-1] + logT[I→I] )          ← leftward predecessor, D→I forbidden
//
//   V_D[i][j] = max( V_M[i-1][j] + logT[M→D],
//                    V_D[i-1][j] + logT[D→D] )          ← upward predecessor, I→D forbidden
//
// BOUNDARY CONDITIONS:
//   (0,0): seeded with LOG_INIT probabilities for all three states.
//   Row 0 (i=0, j>0): only I active — query advancing against empty reference.
//   Col 0 (i>0, j=0): only D active — reference advancing against empty query.
//
// TRACEBACK ENCODING:
//   Direction (DIR_DIAG/UP/LEFT) and predecessor state are packed into one byte:
//     low  4 bits = direction
//     high 4 bits = predecessor state index (S_M/S_I/S_D)
//   Three separate arrays (TB_M, TB_I, TB_D) store one entry per cell per state.
//   During traceback, cur_state selects which array to read at each step.
//
// PROGRESS REPORTING:
//   Prints progress per pair to stdout since this version can take ~60 minutes
//   for large batches. Flush after each pair to avoid buffered output.
// =============================================================================
static void alignmentOnCPU(
    int32_t* d_info,    // [0]=numPairs, [1]=longestLen (stride)
    int32_t* d_seqLen,  // actual length of each sequence
    char*    d_seqs,    // flat packed sequences, stride=longestLen
    uint8_t* d_tb)      // output: traceback direction codes per pair
{
    using namespace HMM;

    const uint8_t DIR_DIAG = 1;   // both sequences advance (match/mismatch)
    const uint8_t DIR_UP   = 2;   // only ref advances (gap in query)
    const uint8_t DIR_LEFT = 3;   // only query advances (gap in ref)

    int32_t numPairs  = d_info[0];
    int32_t maxSeqLen = d_info[1];  // = longestLen, stride between sequences in d_seqs
    int     tb_length = maxSeqLen << 1;

    fprintf(stdout, "Batch started (%d pairs)\n", numPairs);
    fflush(stdout);

    for (int pair = 0; pair < numPairs; ++pair) {

        int32_t refTotalLen = d_seqLen[2 * pair];
        int32_t qryTotalLen = d_seqLen[2 * pair + 1];

        int32_t R = refTotalLen;   // number of ref bases = number of DP rows (1..R)
        int32_t Q = qryTotalLen;   // number of qry bases = number of DP columns (1..Q)

        // Full DP tables: 3 states × (R+1) × (Q+1)
        // Indexed as: state*(R+1)*(Q+1) + i*(Q+1) + j
        // Allocated per-pair since R and Q vary — avoids wasting memory on padding
        int64_t cellCount = (int64_t)(R + 1) * (Q + 1);

        std::vector<double>  V_M(cellCount, NEG_INF);  // Viterbi scores for state M
        std::vector<double>  V_I(cellCount, NEG_INF);  // Viterbi scores for state I
        std::vector<double>  V_D(cellCount, NEG_INF);  // Viterbi scores for state D

        // TB arrays store: direction | (predecessor_state << 4)
        // We pack dir in low 4 bits, pred_state in high 4 bits
        // Reading: dir = tb_byte & 0xF;  prev_state = (tb_byte >> 4) & 0xF;
        std::vector<uint8_t> TB_M(cellCount, 0);  // traceback for state M
        std::vector<uint8_t> TB_I(cellCount, 0);  // traceback for state I
        std::vector<uint8_t> TB_D(cellCount, 0);  // traceback for state D

        // Flat indexer: (i, j) → index in the (R+1)×(Q+1) row-major array
        auto idx = [&](int i, int j) -> int64_t {
            return (int64_t)i * (Q + 1) + j;
        };

        // ------------------------------------------------------------------
        // Initialise (0,0)
        // All three states are seeded with their initial log-probabilities.
        // ------------------------------------------------------------------
        V_M[idx(0, 0)] = LOG_INIT[S_M];
        V_I[idx(0, 0)] = LOG_INIT[S_I];
        V_D[idx(0, 0)] = LOG_INIT[S_D];

        // ------------------------------------------------------------------
        // Top boundary: i=0, j>0  → only I active (query gap)
        // Reference has not started; only query can advance.
        // V_I[0][j] = max(V_M[0][j-1]+logT[M→I], V_I[0][j-1]+logT[I→I])
        // ------------------------------------------------------------------
        for (int j = 1; j <= Q; ++j) {
            // I state: came from M or I at (0, j-1)
            double vMI = V_M[idx(0, j-1)] + LOG_TRANS[S_M][S_I];
            double vII = V_I[idx(0, j-1)] + LOG_TRANS[S_I][S_I];
            if (vMI >= vII) {
                V_I[idx(0, j)] = vMI;
                TB_I[idx(0, j)] = DIR_LEFT | (uint8_t)(S_M << 4);
            } else {
                V_I[idx(0, j)] = vII;
                TB_I[idx(0, j)] = DIR_LEFT | (uint8_t)(S_I << 4);
            }
            // M and D stay NEG_INF on this boundary
        }

        // ------------------------------------------------------------------
        // Left boundary: j=0, i>0  → only D active (ref gap)
        // Query has not started; only reference can advance.
        // V_D[i][0] = max(V_M[i-1][0]+logT[M→D], V_D[i-1][0]+logT[D→D])
        // ------------------------------------------------------------------
        for (int i = 1; i <= R; ++i) {
            double vMD = V_M[idx(i-1, 0)] + LOG_TRANS[S_M][S_D];
            double vDD = V_D[idx(i-1, 0)] + LOG_TRANS[S_D][S_D];
            if (vMD >= vDD) {
                V_D[idx(i, 0)] = vMD;
                TB_D[idx(i, 0)] = DIR_UP | (uint8_t)(S_M << 4);
            } else {
                V_D[idx(i, 0)] = vDD;
                TB_D[idx(i, 0)] = DIR_UP | (uint8_t)(S_D << 4);
            }
            // M and I stay NEG_INF on this boundary
        }

        // ------------------------------------------------------------------
        // Sequence pointers
        // ------------------------------------------------------------------
        int32_t refStart = (pair * 2)     * maxSeqLen;  // start of ref in d_seqs
        int32_t qryStart = (pair * 2 + 1) * maxSeqLen;  // start of qry in d_seqs

        // ------------------------------------------------------------------
        // Fill full DP table row-major
        // For each inner cell (i>0, j>0), compute all three HMM states.
        // ------------------------------------------------------------------
        for (int i = 1; i <= R; ++i) {
            char r = d_seqs[refStart + (i - 1)];  // ref base at position i

            for (int j = 1; j <= Q; ++j) {
                char q = d_seqs[qryStart + (j - 1)];  // qry base at position j
                double logEmit = (r == q) ? LOG_EMIT_MATCH : LOG_EMIT_MISMATCH;

                // ---- State M (diagonal) ----------------------------------------
                // V_M[i][j] = logEmit + max_s(V_s[i-1][j-1] + logT[s→M])
                // All three predecessor states allowed
                {
                    double best = NEG_INF; int bestS = S_M;
                    double vM = V_M[idx(i-1, j-1)] + LOG_TRANS[S_M][S_M];
                    double vI = V_I[idx(i-1, j-1)] + LOG_TRANS[S_I][S_M];
                    double vD = V_D[idx(i-1, j-1)] + LOG_TRANS[S_D][S_M];
                    if (vM >= vI && vM >= vD) { best = vM; bestS = S_M; }
                    else if (vI >= vD)         { best = vI; bestS = S_I; }
                    else                        { best = vD; bestS = S_D; }
                    V_M[idx(i, j)]  = best + logEmit;
                    TB_M[idx(i, j)] = DIR_DIAG | (uint8_t)(bestS << 4);
                }

                // ---- State I (left: query gap) ----------------------------------
                // V_I[i][j] = max(V_M[i][j-1]+logT[M→I], V_I[i][j-1]+logT[I→I])
                // D→I forbidden
                {
                    // D→I forbidden
                    double vMI = V_M[idx(i, j-1)] + LOG_TRANS[S_M][S_I];
                    double vII = V_I[idx(i, j-1)] + LOG_TRANS[S_I][S_I];
                    if (vMI >= vII) {
                        V_I[idx(i, j)]  = vMI;
                        TB_I[idx(i, j)] = DIR_LEFT | (uint8_t)(S_M << 4);
                    } else {
                        V_I[idx(i, j)]  = vII;
                        TB_I[idx(i, j)] = DIR_LEFT | (uint8_t)(S_I << 4);
                    }
                }

                // ---- State D (up: ref gap) --------------------------------------
                // V_D[i][j] = max(V_M[i-1][j]+logT[M→D], V_D[i-1][j]+logT[D→D])
                // I→D forbidden
                {
                    // I→D forbidden
                    double vMD = V_M[idx(i-1, j)] + LOG_TRANS[S_M][S_D];
                    double vDD = V_D[idx(i-1, j)] + LOG_TRANS[S_D][S_D];
                    if (vMD >= vDD) {
                        V_D[idx(i, j)]  = vMD;
                        TB_D[idx(i, j)] = DIR_UP | (uint8_t)(S_M << 4);
                    } else {
                        V_D[idx(i, j)]  = vDD;
                        TB_D[idx(i, j)] = DIR_UP | (uint8_t)(S_D << 4);
                    }
                }
            }
        }

        // ------------------------------------------------------------------
        // Pick best terminal state at (R, Q)
        // The globally optimal alignment ends at (R, Q); select the highest
        // scoring state there to determine the traceback starting state.
        // ------------------------------------------------------------------
        int startState = S_M;
        double bestFinal = V_M[idx(R, Q)];
        if (V_I[idx(R, Q)] > bestFinal) { bestFinal = V_I[idx(R, Q)]; startState = S_I; }
        if (V_D[idx(R, Q)] > bestFinal) {                               startState = S_D; }

        // ------------------------------------------------------------------
        // Traceback from (R, Q) back to (0, 0)
        //
        // At each step, cur_state selects the correct TB array (TB_M/I/D).
        // The packed byte encodes both direction (low 4 bits) and predecessor
        // state (high 4 bits). Directions are collected in reverse into
        // localPath, then written forward into d_tb.
        // ------------------------------------------------------------------
        std::vector<uint8_t> localPath;
        localPath.reserve(R + Q);

        int ti = R, tj = Q;
        int cur_state = startState;

        while (ti > 0 || tj > 0) {
            uint8_t tb_byte;
            uint8_t dir;
            int     prev_state;

            if (ti == 0) {
                // Forced left along top boundary (query gap, state I)
                tb_byte    = TB_I[idx(ti, tj)];
                dir        = DIR_LEFT;
                prev_state = (tb_byte >> 4) & 0xF;  // unpack predecessor from high nibble
                tj--;
            } else if (tj == 0) {
                // Forced up along left boundary (ref gap, state D)
                tb_byte    = TB_D[idx(ti, tj)];
                dir        = DIR_UP;
                prev_state = (tb_byte >> 4) & 0xF;
                ti--;
            } else {
                // Inner cell: select TB array by cur_state, unpack direction and predecessor
                if      (cur_state == S_M) tb_byte = TB_M[idx(ti, tj)];
                else if (cur_state == S_I) tb_byte = TB_I[idx(ti, tj)];
                else                       tb_byte = TB_D[idx(ti, tj)];

                dir        = tb_byte & 0xF;          // low 4 bits = direction
                prev_state = (tb_byte >> 4) & 0xF;  // high 4 bits = predecessor state

                if      (dir == DIR_DIAG) { ti--; tj--; }
                else if (dir == DIR_UP)   { ti--;       }
                else                      {       tj--; }
            }

            localPath.push_back(dir);  // collected in reverse
            cur_state = prev_state;     // follow predecessor state chain
        }

        // ------------------------------------------------------------------
        // Write path forward (reversed) into global traceback buffer
        // ------------------------------------------------------------------
        int32_t tbGlobalOffset = pair * tb_length;
        int32_t pathLen = (int32_t)localPath.size();

        // Guard: should never exceed tb_length, but clamp for safety
        if (pathLen > tb_length) {
            fprintf(stderr, "WARNING: pair %d path length %d exceeds tb_length %d — truncating\n",
                    pair, pathLen, tb_length);
            pathLen = tb_length;
        }

        // Reverse localPath into d_tb (path stored forward: start→end)
        for (int k = pathLen - 1; k >= 0; --k)
            d_tb[tbGlobalOffset + (pathLen - 1 - k)] = localPath[k];

        // Progress report — useful since this can take ~60 minutes for large batches
        fprintf(stdout, "  pair %d / %d in batch done.\n", pair + 1, numPairs);
        fflush(stdout);
    }
    fprintf(stdout, "Batch complete.\n");
    fflush(stdout);
}

// =============================================================================
// getAlignedSequences
//
// Reconstructs gapped alignment strings from the traceback path:
//   DIR_DIAG (1): consume one base from each sequence
//   DIR_UP   (2): consume one ref base, insert '-' into query alignment
//   DIR_LEFT (3): insert '-' into ref alignment, consume one query base
//   0:            end-of-path sentinel — stop
// =============================================================================
void GpuAligner::getAlignedSequences(TB_PATH& tb_paths) {
    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    int tb_length = longestLen << 1;

    for (int pair = 0; pair < numPairs; ++pair) {
        int tb_start = tb_length * pair;
        int seqId0 = 2 * pair, seqId1 = 2 * pair + 1;
        std::string seq0 = seqs[seqId0].seq;
        std::string seq1 = seqs[seqId1].seq;
        std::string aln0, aln1;
        int seqPos0 = 0, seqPos1 = 0;

        for (int i = tb_start; i < tb_start + tb_length; ++i) {
            if      (tb_paths[i] == DIR_DIAG) { aln0 += seq0[seqPos0++]; aln1 += seq1[seqPos1++]; }
            else if (tb_paths[i] == DIR_UP)   { aln0 += seq0[seqPos0++]; aln1 += '-'; }
            else if (tb_paths[i] == DIR_LEFT) { aln0 += '-'; aln1 += seq1[seqPos1++]; }
            else break;  // DIR=0 sentinel: end of path
        }

        seqs[seqId0].aln = aln0;
        seqs[seqId1].aln = aln1;
    }
}

// =============================================================================
// clearAndReset
//
// Frees all heap-allocated buffers and resets state for the next batch.
// =============================================================================
void GpuAligner::clearAndReset() {
    delete[] d_seqs;
    delete[] d_seqLen;
    delete[] d_tb;
    delete[] d_info;
    seqs.clear();
    longestLen = 0;
    numPairs   = 0;
}

// =============================================================================
// alignment
//
// Top-level orchestration: allocate → pack sequences → run full Viterbi DP →
// extract traceback → reconstruct aligned strings.
// =============================================================================
void GpuAligner::alignment() {
    allocateMem();
    transferSequence2Device();
    alignmentOnCPU(d_info, d_seqLen, d_seqs, d_tb);
    TB_PATH tb_paths = transferTB2Host();
    getAlignedSequences(tb_paths);
}

// =============================================================================
// writeAlignment  — writes to hmm_reference.fa (or any given filename)
//
// Output is used as the ground-truth reference for compare_alignment.
// =============================================================================
void GpuAligner::writeAlignment(std::string fileName, bool append) {
    std::ofstream outFile;
    if (append) outFile.open(fileName, std::ios::app);
    else        outFile.open(fileName);
    if (!outFile) { fprintf(stderr, "ERROR: cant open file: %s\n", fileName.c_str()); exit(1); }
    for (auto& seq : seqs) {
        outFile << '>' << seq.name << '\n';
        outFile << seq.aln        << '\n';
    }
    outFile.close();
}
