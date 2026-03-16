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
// This file implements Viterbi decoding for a Pair Hidden Markov Model (Pair-HMM)
// combined with the GACT tiling heuristic to handle long sequences efficiently.
//
// The Pair-HMM has three hidden states, each corresponding to an alignment
// operation between a reference sequence (ref) and a query sequence (qry):
//
//   M (0) – Match / Mismatch  → both sequences advance one base  → DIR_DIAG
//   I (1) – Insert in query   → only query advances (ref gap)    → DIR_LEFT
//   D (2) – Delete from query → only ref advances  (query gap)   → DIR_UP
//
// Only state M emits a pair of observed bases. States I and D are "silent"
// with respect to one of the two sequences.
//
// Emission probabilities (state M only):
//   p(match)    = 0.90   — two aligned bases are identical
//   p(mismatch) = 0.10   — two aligned bases differ
//
// Transition matrix (row = from-state, col = to-state):
//   M → { M:0.90,  I:0.05,  D:0.05 }
//   I → { M:0.70,  I:0.30,  D:---  }   I→D forbidden (prevents alignment ambiguity)
//   D → { M:0.70,  I:---,   D:0.30 }   D→I forbidden (prevents alignment ambiguity)
//
//   The I→D and D→I transitions are forbidden because allowing them would create
//   multiple equally-scored representations of the same alignment (e.g., a
//   deletion followed by an insertion could be scored identically to the reverse).
//
// Initial state probabilities (probability of starting in each state):
//   π(M)=0.90, π(I)=0.05, π(D)=0.05
//
// These parameters implicitly encode affine gap penalties:
//   - Gap open cost   ≈ log(0.05) - log(0.90) ≈ -2.89 nats  (expensive)
//   - Gap extend cost ≈ log(0.30) - log(0.70) ≈ -0.85 nats  (cheaper than opening)
//
// All values are stored as natural logarithms to avoid floating-point underflow
// when multiplying many small probabilities over long sequences. Addition in
// log-space replaces multiplication in probability space.
// =============================================================================

namespace HMM {
    // Sentinel value representing -infinity in log-space (probability = 0)
    static const double NEG_INF = -1e18;

    // Log-emission probabilities for state M
    static const double LOG_EMIT_MATCH    = std::log(0.90);
    static const double LOG_EMIT_MISMATCH = std::log(0.10);

    // LOG_TRANS[from][to] — log transition probability from state 'from' to state 'to'
    // NEG_INF entries are forbidden transitions (probability = 0)
    static const double LOG_TRANS[3][3] = {
        { std::log(0.90), std::log(0.05), std::log(0.05) }, // from M: M→M, M→I, M→D
        { std::log(0.70), std::log(0.30), NEG_INF         }, // from I: I→M, I→I, I→D(forbidden)
        { std::log(0.70), NEG_INF,        std::log(0.30)  }  // from D: D→M, D→I(forbidden), D→D
    };

    // LOG_INIT[s] — log probability of starting the alignment in state s
    static const double LOG_INIT[3] = {
        std::log(0.90),  // start in M (most likely)
        std::log(0.05),  // start in I
        std::log(0.05)   // start in D
    };

    // State index constants for readability
    static const int S_M = 0;
    static const int S_I = 1;
    static const int S_D = 2;
}

// =============================================================================
// allocateMem
//
// Allocates flat CPU arrays that mirror the GPU memory layout used in the
// GPU implementations. This allows the same alignment and traceback logic
// to be shared between CPU and GPU versions.
//
// Memory layout for d_seqs:
//   [ref_0 | pad | qry_0 | pad | ref_1 | pad | qry_1 | pad | ...]
//   Each sequence occupies exactly 'longestLen' bytes (zero-padded).
//   Sequence i is accessed as: d_seqs + i * longestLen
//
// Memory layout for d_tb (traceback buffer):
//   Each pair gets 2*longestLen bytes (worst-case path length is all gaps).
//   Pair p starts at: d_tb + p * 2 * longestLen
// =============================================================================
void GpuAligner::allocateMem() {
    // Find the longest sequence to determine the stride for the flat array
    longestLen = std::max_element(seqs.begin(), seqs.end(), [](const Sequence& a, const Sequence& b) {
        return a.seq.size() < b.seq.size();
    })->seq.size();

    // Flat array holding all reference and query sequences, zero-padded to longestLen
    d_seqs   = new char   [numPairs * 2 * longestLen]();
    if (!d_seqs)   { fprintf(stderr, "CPU_ERROR: alloc d_seqs\n");   exit(1); }

    // Array of actual sequence lengths (needed to avoid processing padding zeros)
    d_seqLen = new int32_t[numPairs * 2]();
    if (!d_seqLen) { fprintf(stderr, "CPU_ERROR: alloc d_seqLen\n"); exit(1); }

    // Traceback path buffer — worst case 2*longestLen directions per pair
    int tb_length = longestLen << 1;
    d_tb     = new uint8_t [numPairs * tb_length]();
    if (!d_tb)     { fprintf(stderr, "CPU_ERROR: alloc d_tb\n");     exit(1); }

    // Two-element info array: [0] = numPairs, [1] = longestLen (stride)
    d_info   = new int32_t[2]();
    if (!d_info)   { fprintf(stderr, "CPU_ERROR: alloc d_info\n");   exit(1); }
}

// =============================================================================
// transferSequence2Device  (host→host, no GPU)
//
// Copies sequence data from the structured Sequence objects into the flat
// C-style arrays expected by the alignment kernel. This function is named
// "transferSequence2Device" for API compatibility with the GPU versions,
// but on CPU it is purely a memory-to-memory copy with no device transfer.
//
// After this call:
//   d_seqs[i * longestLen .. i * longestLen + len_i - 1] = sequence i's bases
//   d_seqLen[i]  = actual length of sequence i (excluding padding)
//   d_info[0]    = numPairs
//   d_info[1]    = longestLen (the stride between sequences in d_seqs)
// =============================================================================
void GpuAligner::transferSequence2Device() {
    // Pack each sequence into the flat array at its strided offset
    for (size_t i = 0; i < numPairs * 2; ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(d_seqs + (i * longestLen), s.data(), s.size());
    }
    // Record actual lengths so the kernel knows where valid data ends
    for (int i = 0; i < (int)(numPairs * 2); ++i)
        d_seqLen[i] = (int32_t)seqs[i].seq.size();

    d_info[0] = numPairs;
    d_info[1] = longestLen;
}

// =============================================================================
// transferTB2Host
//
// Copies the computed traceback paths back from the flat d_tb buffer into
// a std::vector for use by getAlignedSequences. On CPU this is a plain
// memcpy; on the GPU versions this is a cudaMemcpy DeviceToHost.
// =============================================================================
TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen << 1;
    TB_PATH h_tb(tb_length * numPairs);
    std::memcpy(h_tb.data(), d_tb, tb_length * numPairs * sizeof(uint8_t));
    return h_tb;
}

// =============================================================================
// alignmentOnCPU  —  Viterbi Pair-HMM with GACT tiling
// =============================================================================
//
// Computes the Viterbi (maximum-likelihood) alignment for each sequence pair
// using the Pair-HMM model defined above, with the GACT tiling heuristic to
// keep memory usage bounded for long sequences.
//
// --- VITERBI RECURRENCES (1-based indices inside each tile) ---
//
// Three DP tables are maintained, one per HMM state. Each cell (i,j) stores
// the log-probability of the best partial alignment ending at ref[i], qry[j]
// while in that state:
//
//   V_M[i][j] = logEmit(r[i], q[j])
//              + max_s( V_s[i-1][j-1] + logT[s→M] )
//              — diagonal move; best predecessor across all three states
//
//   V_I[i][j] = max( V_M[i][j-1] + logT[M→I],
//                    V_I[i][j-1] + logT[I→I] )
//              — leftward move (gap in ref); only M and I can precede I
//
//   V_D[i][j] = max( V_M[i-1][j] + logT[M→D],
//                    V_D[i-1][j] + logT[D→D] )
//              — upward move (gap in query); only M and D can precede D
//
// Row 0 (j=0) and column 0 (i=0) are boundary conditions handled separately.
//
// --- TRACEBACK STORAGE ---
//
// For each (state, i, j), two values are stored:
//   TB_dir[state][i][j]   — direction of the move that reached this cell
//                           (DIR_DIAG for M, DIR_LEFT for I, DIR_UP for D)
//   TB_state[state][i][j] — which predecessor state produced the best score
//
// During traceback, cur_state determines which table to read, and TB_state
// gives the next cur_state. This is necessary because different predecessor
// states can reach the same (i,j) cell via different paths.
//
// --- GACT TILING ---
//
// For sequences longer than T bases, the full DP matrix would require O(mn)
// memory. GACT tiles the alignment into T×T blocks processed sequentially.
// At the end of each non-final tile, the best-scoring cell in the bottom-right
// O×O corner (the "overlap region") is identified. Its score (carryLogProb)
// and state (carryState) seed the next tile at position (0,0), allowing the
// alignment path to shift relative to the diagonal between tiles.
//
// Parameters: T=200 (tile size), O=64 (overlap region size)
// =============================================================================
static void alignmentOnCPU(
    int32_t* d_info,    // [0]=numPairs, [1]=longestLen (stride)
    int32_t* d_seqLen,  // actual length of each sequence
    char*    d_seqs,    // flat packed sequence array
    uint8_t* d_tb)      // output: traceback direction codes per pair
{
    using namespace HMM;

    const int T = 200;   // tile size: process at most T ref bases × T query bases per tile
    const int O = 64;    // overlap size: scan the last O×O corner to find the next tile seed

    // Traceback direction codes written into d_tb
    const uint8_t DIR_DIAG = 1;   // match/mismatch: both sequences advance
    const uint8_t DIR_UP   = 2;   // deletion: only ref advances (gap in query)
    const uint8_t DIR_LEFT = 3;   // insertion: only query advances (gap in ref)

    // DP tables flattened as [state][i][j] — state ∈ {M,I,D}, i ∈ [0,T], j ∈ [0,T]
    // Using (T+1)×(T+1) to accommodate the boundary row (i=0) and column (j=0)
    const int CELL = (T + 1) * (T + 1);
    std::vector<double>  V       (3 * CELL, NEG_INF);  // Viterbi scores
    std::vector<uint8_t> TB_dir  (3 * CELL, 0);        // traceback directions
    std::vector<uint8_t> TB_state(3 * CELL, 0);        // traceback predecessor states

    // Temporary buffer for one tile's traceback path (collected in reverse, then reversed)
    std::vector<uint8_t> localPath(2 * T);

    int32_t numPairs  = d_info[0];
    int32_t maxSeqLen = d_info[1];  // stride between sequences in d_seqs

    // Convenience indexer: converts (state, i, j) → flat array index
    // state * CELL + i * (T+1) + j
    auto idx = [&](int state, int i, int j) -> int {
        return state * CELL + i * (T + 1) + j;
    };

    // -----------------------------------------------------------------------
    // PAIR LOOP — process each pair of sequences independently
    // -----------------------------------------------------------------------
    for (int pair = 0; pair < numPairs; ++pair) {

        bool    lastTile           = false;  // true when the current tile covers the end of both sequences
        int32_t currentPairPathLen = 0;      // how many direction codes written so far for this pair
        int32_t reference_idx      = 0;      // current position in the reference (tile start offset)
        int32_t query_idx          = 0;      // current position in the query (tile start offset)

        // carryLogProb: the Viterbi score of the best overlap cell from the previous tile.
        // It seeds V[s][0][0] = carryLogProb + LOG_INIT[s] at the start of each new tile.
        // Initialised to 0.0 = log(1) — neutral probability for the very first tile.
        double carryLogProb = 0.0;
        int    carryState   = S_M;  // HMM state at the carry-in cell (unused in current seeding)

        // Byte offsets into the flat sequence and traceback arrays for this pair
        int32_t refStart       = (pair * 2)     * maxSeqLen;   // d_seqs[refStart] = ref seq start
        int32_t qryStart       = (pair * 2 + 1) * maxSeqLen;   // d_seqs[qryStart] = qry seq start
        int32_t tbGlobalOffset = pair * (maxSeqLen * 2);        // d_tb[tbGlobalOffset] = path start

        int32_t refTotalLen = d_seqLen[2 * pair];       // actual reference length
        int32_t qryTotalLen = d_seqLen[2 * pair + 1];   // actual query length

        // -------------------------------------------------------------------
        // TILE LOOP — iterate over T×T tiles along the alignment path
        // -------------------------------------------------------------------
        while (!lastTile) {
            // Compute how many bases are left in each sequence for this tile
            // (may be less than T for the final tile)
            int32_t refLen = std::min(T, (int)(refTotalLen - reference_idx));
            int32_t qryLen = std::min(T, (int)(qryTotalLen - query_idx));

            // Mark as last tile if this tile reaches the end of both sequences
            if ((reference_idx + refLen == refTotalLen) &&
                (query_idx     + qryLen == qryTotalLen))
                lastTile = true;

            // ------------------------------------------------------------------
            // INITIALISE DP TABLES for this tile
            // All cells start at NEG_INF (log(0) — unreachable)
            // ------------------------------------------------------------------
            std::fill(V.begin(),        V.end(),        NEG_INF);
            std::fill(TB_dir.begin(),   TB_dir.end(),   0);
            std::fill(TB_state.begin(), TB_state.end(), 0);

            // Cell (0,0) is the tile origin. Seed all three states from the
            // carry-in score of the previous tile's best overlap cell.
            // For the first tile, carryLogProb=0 so V[s][0][0] = LOG_INIT[s].
            for (int s = 0; s < 3; ++s)
                V[idx(s, 0, 0)] = carryLogProb + LOG_INIT[s];

            // Top boundary (i > 0, j = 0): the query has not advanced at all,
            // so only state D (deletion / gap-in-query) is reachable.
            // Each cell inherits from the best predecessor state above it.
            for (int i = 1; i <= refLen; ++i) {
                double best = NEG_INF; int bestS = S_M;
                for (int s = 0; s < 3; ++s) {
                    double v = V[idx(s, i - 1, 0)] + LOG_TRANS[s][S_D];
                    if (v > best) { best = v; bestS = s; }
                }
                V[idx(S_D, i, 0)]        = best;
                TB_dir  [idx(S_D, i, 0)] = DIR_UP;
                TB_state[idx(S_D, i, 0)] = (uint8_t)bestS;
            }

            // Left boundary (i = 0, j > 0): the reference has not advanced,
            // so only state I (insertion / gap-in-ref) is reachable.
            for (int j = 1; j <= qryLen; ++j) {
                double best = NEG_INF; int bestS = S_M;
                for (int s = 0; s < 3; ++s) {
                    double v = V[idx(s, 0, j - 1)] + LOG_TRANS[s][S_I];
                    if (v > best) { best = v; bestS = s; }
                }
                V[idx(S_I, 0, j)]        = best;
                TB_dir  [idx(S_I, 0, j)] = DIR_LEFT;
                TB_state[idx(S_I, 0, j)] = (uint8_t)bestS;
            }

            // ------------------------------------------------------------------
            // FILL DP TABLE — row-major traversal of inner cells (i≥1, j≥1)
            // Track the best-scoring cell in the overlap region for GACT.
            // ------------------------------------------------------------------
            double bestOverlapScore = NEG_INF;
            int    best_state = S_M, best_ti = refLen, best_tj = qryLen;

            for (int i = 1; i <= refLen; ++i) {
                char r = d_seqs[refStart + reference_idx + (i - 1)];  // ref base at position i

                for (int j = 1; j <= qryLen; ++j) {
                    char q = d_seqs[qryStart + query_idx + (j - 1)];  // qry base at position j

                    // Log-emission: log p(r,q | state M)
                    double logEmit = (r == q) ? LOG_EMIT_MATCH : LOG_EMIT_MISMATCH;

                    // ---- State M: diagonal move, best predecessor from all 3 states ----
                    // V_M[i][j] = logEmit + max_s( V_s[i-1][j-1] + logT[s→M] )
                    {
                        double best = NEG_INF; int bestS = S_M;
                        for (int s = 0; s < 3; ++s) {
                            double v = V[idx(s, i - 1, j - 1)] + LOG_TRANS[s][S_M];
                            if (v > best) { best = v; bestS = s; }
                        }
                        V[idx(S_M, i, j)]        = best + logEmit;
                        TB_dir  [idx(S_M, i, j)] = DIR_DIAG;
                        TB_state[idx(S_M, i, j)] = (uint8_t)bestS;
                    }

                    // ---- State I: leftward move, only M and I can precede (D→I forbidden) ----
                    // V_I[i][j] = max( V_M[i][j-1] + logT[M→I], V_I[i][j-1] + logT[I→I] )
                    {
                        double best = NEG_INF; int bestS = S_M;
                        for (int s : {S_M, S_I}) {   // D→I forbidden
                            double v = V[idx(s, i, j - 1)] + LOG_TRANS[s][S_I];
                            if (v > best) { best = v; bestS = s; }
                        }
                        V[idx(S_I, i, j)]        = best;
                        TB_dir  [idx(S_I, i, j)] = DIR_LEFT;
                        TB_state[idx(S_I, i, j)] = (uint8_t)bestS;
                    }

                    // ---- State D: upward move, only M and D can precede (I→D forbidden) ----
                    // V_D[i][j] = max( V_M[i-1][j] + logT[M→D], V_D[i-1][j] + logT[D→D] )
                    {
                        double best = NEG_INF; int bestS = S_M;
                        for (int s : {S_M, S_D}) {   // I→D forbidden
                            double v = V[idx(s, i - 1, j)] + LOG_TRANS[s][S_D];
                            if (v > best) { best = v; bestS = s; }
                        }
                        V[idx(S_D, i, j)]        = best;
                        TB_dir  [idx(S_D, i, j)] = DIR_UP;
                        TB_state[idx(S_D, i, j)] = (uint8_t)bestS;
                    }

                    // ---- GACT overlap tracking --------------------------------
                    // For non-final tiles, scan the bottom-right O×O corner.
                    // The cell with the highest score across all three states
                    // becomes the seed (origin) for the next tile.
                    // Using >= so that on equal scores the last state wins (D > I > M),
                    // consistent with the GPU parallel reduction tie-breaking.
                    if (!lastTile &&
                        i > (refLen - O) && j > (qryLen - O)) {
                        for (int s = 0; s < 3; ++s) {
                            if (V[idx(s, i, j)] >= bestOverlapScore) {
                                bestOverlapScore = V[idx(s, i, j)];
                                best_state = s;
                                best_ti    = i;
                                best_tj    = j;
                            }
                        }
                    }
                }
            } // end fill

            // ------------------------------------------------------------------
            // DETERMINE TRACEBACK START and update carry-in for next tile
            // ------------------------------------------------------------------
            int ti, tj, startState;
            if (!lastTile) {
                // Non-final tile: start traceback from the best overlap cell
                ti = best_ti; tj = best_tj; startState = best_state;
                // Carry this score and state into the next tile's (0,0) origin
                carryLogProb = bestOverlapScore;
                carryState   = best_state;
            } else {
                // Final tile: start traceback from the terminal cell (refLen, qryLen)
                // and pick the state with the highest score at that cell
                ti = refLen; tj = qryLen;
                carryLogProb = NEG_INF; startState = S_M;
                for (int s = 0; s < 3; ++s) {
                    if (V[idx(s, ti, tj)] > carryLogProb) {
                        carryLogProb = V[idx(s, ti, tj)];
                        startState   = s;
                    }
                }
                carryState = startState;
            }

            // How far to advance the sequence cursors after writing this tile's path
            int next_ref_advance = ti;
            int next_qry_advance = tj;

            // ------------------------------------------------------------------
            // TRACEBACK — walk backwards from (ti, tj) to (0, 0)
            //
            // At each step, cur_state tells us which DP table to read.
            // TB_dir gives the direction (and thus how to decrement ti/tj).
            // TB_state gives the predecessor state for the next step.
            // The path is collected in reverse order into localPath[].
            // ------------------------------------------------------------------
            int localLen  = 0;
            int cur_state = startState;

            while (ti > 0 || tj > 0) {
                uint8_t dir;
                int     prev_state;

                if (ti == 0) {
                    // Top boundary: forced leftward (query-only gap, state I)
                    dir        = DIR_LEFT;
                    prev_state = TB_state[idx(S_I, ti, tj)];
                    tj--;
                } else if (tj == 0) {
                    // Left boundary: forced upward (ref-only gap, state D)
                    dir        = DIR_UP;
                    prev_state = TB_state[idx(S_D, ti, tj)];
                    ti--;
                } else {
                    // Inner cell: follow the stored direction for cur_state
                    dir        = TB_dir  [idx(cur_state, ti, tj)];
                    prev_state = TB_state[idx(cur_state, ti, tj)];
                    if      (dir == DIR_DIAG) { ti--; tj--; }
                    else if (dir == DIR_UP)   { ti--;        }
                    else                      {       tj--;  }
                }

                localPath[localLen++] = dir;  // store in reverse
                cur_state = prev_state;        // step to predecessor state
            }

            // Reverse localPath into d_tb (traceback buffer is stored forward: start→end)
            for (int k = localLen - 1; k >= 0; --k) {
                d_tb[tbGlobalOffset + currentPairPathLen] = localPath[k];
                currentPairPathLen++;
            }

            // Advance sequence cursors to the start of the next tile
            reference_idx += next_ref_advance;
            query_idx     += next_qry_advance;

        } // end tile loop
    } // end pair loop
}

// =============================================================================
// getAlignedSequences
//
// Reconstructs the actual aligned strings from the compact traceback path
// stored in tb_paths. Each direction code is interpreted as:
//   DIR_DIAG (1): both sequences emit one base → consume seq0[seqPos0], seq1[seqPos1]
//   DIR_UP   (2): ref gap   → consume seq0[seqPos0], insert '-' into aln1
//   DIR_LEFT (3): query gap → insert '-' into aln0, consume seq1[seqPos1]
//   0 / other:    end of path — stop
// =============================================================================
void GpuAligner::getAlignedSequences(TB_PATH& tb_paths) {
    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    int tb_length = longestLen << 1;  // allocated path length per pair

    for (int pair = 0; pair < numPairs; ++pair) {
        int tb_start = tb_length * pair;   // start of this pair's path in tb_paths
        int seqId0 = 2 * pair, seqId1 = 2 * pair + 1;
        std::string seq0 = seqs[seqId0].seq;
        std::string seq1 = seqs[seqId1].seq;
        std::string aln0, aln1;
        int seqPos0 = 0, seqPos1 = 0;

        for (int i = tb_start; i < tb_start + tb_length; ++i) {
            if      (tb_paths[i] == DIR_DIAG) { aln0 += seq0[seqPos0++]; aln1 += seq1[seqPos1++]; }
            else if (tb_paths[i] == DIR_UP)   { aln0 += seq0[seqPos0++]; aln1 += '-'; }
            else if (tb_paths[i] == DIR_LEFT) { aln0 += '-'; aln1 += seq1[seqPos1++]; }
            else break;  // 0 = end of path sentinel
        }

        seqs[seqId0].aln = aln0;
        seqs[seqId1].aln = aln1;
    }
}

// =============================================================================
// clearAndReset
//
// Frees all heap-allocated buffers and resets the aligner state so the
// same GpuAligner object can be reused for the next batch of sequences.
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
// Top-level orchestration: allocate → pack sequences → run Viterbi →
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
// writeAlignment
//
// Writes all aligned sequences to a FASTA file. If append=true, the file is
// opened in append mode (used when writing multiple batches to the same output
// file). Otherwise the file is overwritten from the start.
// =============================================================================
void GpuAligner::writeAlignment(std::string fileName, bool append) {
    std::ofstream outFile;
    if (append) outFile.open(fileName, std::ios::app);
    else        outFile.open(fileName);
    if (!outFile) { fprintf(stderr, "ERROR: cant open file: %s\n", fileName.c_str()); exit(1); }
    for (auto& seq : seqs) {
        outFile << ('>' + seq.name + '\n');
        outFile << (seq.aln + '\n');
    }
    outFile.close();
}
