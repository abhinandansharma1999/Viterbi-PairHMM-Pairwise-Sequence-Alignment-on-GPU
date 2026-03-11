/*
 * ============================================================================
 * HMM-BASED PAIRWISE SEQUENCE ALIGNMENT — CPU IMPLEMENTATION
 * ============================================================================
 *
 * HIDDEN MARKOV MODEL: Pair HMM (Durbin et al., 1998)
 * ====================================================
 * Pure C++ CPU port of the CUDA single-thread kernel.
 * Uses the SAME GpuAligner struct and ALL the same function signatures
 * declared in alignment.cuh — no header changes required.
 *
 * CUDA-specific struct members (d_seqs, d_tb, d_seqLen, d_info) are
 * declared in the header but unused here; they are set to nullptr.
 * All data stays on the CPU heap via std::vector.
 *
 *   States:
 *     M  = Match/Mismatch  (both sequences advance)
 *     IX = Insert in X     (query advances, gap in ref)
 *     IY = Insert in Y     (ref advances,   gap in query)
 *
 *   Transition Probabilities (log-space, nats):
 *   From \ To     M           IX          IY
 *   M          log(0.80)   log(0.10)   log(0.10)
 *   IX         log(0.15)   log(0.80)   log(0.05)
 *   IY         log(0.15)   log(0.05)   log(0.80)
 *
 *   Emission Probabilities:
 *     e_M(x==y) = ln(0.85)   (match)
 *     e_M(x!=y) = ln(0.05)   (mismatch)
 *     e_I(any)  = ln(0.25)   (gap, uniform over alphabet)
 *
 *   Effective Viterbi Scores (scaled x5, int16_t):
 *     MATCH      =  +8
 *     MISMATCH   =  -15
 *     GAP_OPEN_M =  -19   (M  -> IX or IY)
 *     GAP_EXT    =   -8   (IX -> IX or IY -> IY)
 *     CLOSE_GAP  =  -10   (IX -> M  or IY -> M)
 *     SWITCH_GAP =  -22   (IX -> IY or IY -> IX)
 *
 * TILING STRATEGY (GACT-style, T=10, O=3):
 *   Tile by tile; best M-state score in overlap region seeds next tile.
 * ============================================================================
 */

#include "alignment.cuh"   // unchanged — same header as GPU version
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

using std::max;
using std::min;

// ============================================================================
// Scoring constants (identical to the CUDA kernel, prefixed to avoid ODR issues)
// ============================================================================
static const int16_t CPU_MATCH      =  8;
static const int16_t CPU_MISMATCH   = -15;
static const int16_t CPU_GAP_OPEN_M = -19;
static const int16_t CPU_GAP_EXT    =  -8;
static const int16_t CPU_CLOSE_GAP  = -10;
static const int16_t CPU_SWITCH_GAP = -22;
static const int16_t CPU_NEG_INF    = -9999;

static const uint8_t CPU_DIR_DIAG = 1;
static const uint8_t CPU_DIR_UP   = 2;
static const uint8_t CPU_DIR_LEFT = 3;

// ============================================================================
// alignOnePair (internal helper — not declared in the header)
// ------------
// CPU equivalent of the CUDA kernel body for a single sequence pair.
// Uses the same flat-array layout as the GPU version so scoring logic is
// a direct translation with __shared__ -> stack arrays, no __syncthreads.
//
//   seqs        — flat char array (stride = maxSeqLen)
//   refStart    — index into seqs for start of reference
//   qryStart    — index into seqs for start of query
//   refTotalLen — true (unpadded) reference length
//   qryTotalLen — true (unpadded) query length
//   tb          — output traceback buffer (pre-zeroed, length = tb_length)
//   tb_length   — allocated size of tb
// ============================================================================
static void alignOnePair(
    const char* seqs,
    int32_t     refStart,
    int32_t     qryStart,
    int32_t     refTotalLen,
    int32_t     qryTotalLen,
    uint8_t*    tb,
    int32_t     tb_length
) {
    const int T = 200;
    const int O = 64;

    // Stack arrays replace CUDA __shared__ memory
    int16_t wf_M  [3 * (T + 1)];
    int16_t wf_IX [3 * (T + 1)];
    int16_t wf_IY [3 * (T + 1)];
    uint8_t tbDir [T * T];
    uint8_t localPath[2 * T];

    bool    lastTile       = false;
    int16_t maxScore       = 0;      // seeds first tile's (0,0) cell
    int32_t currentPathLen = 0;
    int32_t reference_idx  = 0;
    int32_t query_idx      = 0;

    // =========================================================================
    // TILE LOOP
    // =========================================================================
    while (!lastTile) {

        int32_t refLen = min((int32_t)T, refTotalLen - reference_idx);
        int32_t qryLen = min((int32_t)T, qryTotalLen - query_idx);

        if ((reference_idx + refLen == refTotalLen) &&
            (query_idx     + qryLen == qryTotalLen)) {
            lastTile = true;
        }

        // Initialize wavefront buffers to NEG_INF
        for (int s = 0; s < 3 * (T + 1); ++s) {
            wf_M [s] = CPU_NEG_INF;
            wf_IX[s] = CPU_NEG_INF;
            wf_IY[s] = CPU_NEG_INF;
        }

        int best_ti = refLen;
        int best_tj = qryLen;

        // =====================================================================
        // WAVEFRONT SCORING LOOP (anti-diagonal k = i + j)
        // =====================================================================
        for (int k = 0; k <= refLen + qryLen; ++k) {

            int curr_k   = (k % 3)      * (T + 1);
            int pre_k    = ((k + 2) % 3) * (T + 1);
            int prepre_k = ((k + 1) % 3) * (T + 1);

            int i_start = max(0, k - (int)qryLen);
            int i_end   = min((int)refLen, k);

            for (int i = i_start; i <= i_end; ++i) {
                int j = k - i;

                int16_t vm  = CPU_NEG_INF;
                int16_t vix = CPU_NEG_INF;
                int16_t viy = CPU_NEG_INF;
                uint8_t dir = CPU_DIR_DIAG;

                // -------------------------------------------------------------
                // Boundary / initialization conditions
                // -------------------------------------------------------------
                if (i == 0 && j == 0) {
                    // Seed M state from previous tile's best M-score.
                    // maxScore reset here (at the point of consumption) so
                    // this tile's overlap tracking starts fresh.
                    vm       = maxScore;
                    maxScore = CPU_NEG_INF;
                    vix      = CPU_NEG_INF;
                    viy      = CPU_NEG_INF;
                }
                else if (i == 0) {
                    // Top edge: only IX reachable (gap in ref)
                    int16_t from_m  = (wf_M [pre_k + i] == CPU_NEG_INF) ? CPU_NEG_INF
                                      : (int16_t)(wf_M [pre_k + i] + CPU_GAP_OPEN_M);
                    int16_t from_ix = (wf_IX[pre_k + i] == CPU_NEG_INF) ? CPU_NEG_INF
                                      : (int16_t)(wf_IX[pre_k + i] + CPU_GAP_EXT);
                    vix = max(from_m, from_ix);
                    vm  = CPU_NEG_INF;
                    viy = CPU_NEG_INF;
                    dir = CPU_DIR_LEFT;
                }
                else if (j == 0) {
                    // Left edge: only IY reachable (gap in query)
                    int16_t from_m  = (wf_M [pre_k + (i-1)] == CPU_NEG_INF) ? CPU_NEG_INF
                                      : (int16_t)(wf_M [pre_k + (i-1)] + CPU_GAP_OPEN_M);
                    int16_t from_iy = (wf_IY[pre_k + (i-1)] == CPU_NEG_INF) ? CPU_NEG_INF
                                      : (int16_t)(wf_IY[pre_k + (i-1)] + CPU_GAP_EXT);
                    viy = max(from_m, from_iy);
                    vm  = CPU_NEG_INF;
                    vix = CPU_NEG_INF;
                    dir = CPU_DIR_UP;
                }
                else {
                    // ---------------------------------------------------------
                    // Inner cell (i > 0, j > 0): full Pair HMM Viterbi step
                    // ---------------------------------------------------------
                    char r_char = seqs[refStart + reference_idx + (i - 1)];
                    char q_char = seqs[qryStart + query_idx     + (j - 1)];

                    int16_t emit_m = (r_char == q_char) ? CPU_MATCH : CPU_MISMATCH;

                    // ---- State M ----
                    // VM[i][j] = emit_m + max( VM[i-1][j-1],
                    //                          VIX[i-1][j-1] + CLOSE_GAP,
                    //                          VIY[i-1][j-1] + CLOSE_GAP )
                    int16_t vm_from_m  = (wf_M [prepre_k + (i-1)] == CPU_NEG_INF) ? CPU_NEG_INF
                                         : (int16_t)(wf_M [prepre_k + (i-1)] + emit_m);
                    int16_t vm_from_ix = (wf_IX[prepre_k + (i-1)] == CPU_NEG_INF) ? CPU_NEG_INF
                                         : (int16_t)(wf_IX[prepre_k + (i-1)] + CPU_CLOSE_GAP + emit_m);
                    int16_t vm_from_iy = (wf_IY[prepre_k + (i-1)] == CPU_NEG_INF) ? CPU_NEG_INF
                                         : (int16_t)(wf_IY[prepre_k + (i-1)] + CPU_CLOSE_GAP + emit_m);
                    vm  = vm_from_m;
                    dir = CPU_DIR_DIAG;
                    if (vm_from_ix > vm) { vm = vm_from_ix; dir = CPU_DIR_LEFT; }
                    if (vm_from_iy > vm) { vm = vm_from_iy; dir = CPU_DIR_UP;   }

                    // ---- State IX (gap in ref / LEFT) ----
                    // VIX[i][j] = max( VM[i][j-1]  + GAP_OPEN_M,
                    //                  VIX[i][j-1] + GAP_EXT,
                    //                  VIY[i][j-1] + SWITCH_GAP )
                    int16_t vix_from_m  = (wf_M [pre_k + i] == CPU_NEG_INF) ? CPU_NEG_INF
                                          : (int16_t)(wf_M [pre_k + i] + CPU_GAP_OPEN_M);
                    int16_t vix_from_ix = (wf_IX[pre_k + i] == CPU_NEG_INF) ? CPU_NEG_INF
                                          : (int16_t)(wf_IX[pre_k + i] + CPU_GAP_EXT);
                    int16_t vix_from_iy = (wf_IY[pre_k + i] == CPU_NEG_INF) ? CPU_NEG_INF
                                          : (int16_t)(wf_IY[pre_k + i] + CPU_SWITCH_GAP);
                    vix = max(vix_from_m, max(vix_from_ix, vix_from_iy));

                    // ---- State IY (gap in query / UP) ----
                    // VIY[i][j] = max( VM[i-1][j]  + GAP_OPEN_M,
                    //                  VIX[i-1][j] + SWITCH_GAP,
                    //                  VIY[i-1][j] + GAP_EXT )
                    int16_t viy_from_m  = (wf_M [pre_k + (i-1)] == CPU_NEG_INF) ? CPU_NEG_INF
                                          : (int16_t)(wf_M [pre_k + (i-1)] + CPU_GAP_OPEN_M);
                    int16_t viy_from_ix = (wf_IX[pre_k + (i-1)] == CPU_NEG_INF) ? CPU_NEG_INF
                                          : (int16_t)(wf_IX[pre_k + (i-1)] + CPU_SWITCH_GAP);
                    int16_t viy_from_iy = (wf_IY[pre_k + (i-1)] == CPU_NEG_INF) ? CPU_NEG_INF
                                          : (int16_t)(wf_IY[pre_k + (i-1)] + CPU_GAP_EXT);
                    viy = max(viy_from_m, max(viy_from_ix, viy_from_iy));
                }

                // Write scores into wavefront buffers
                wf_M [curr_k + i] = vm;
                wf_IX[curr_k + i] = vix;
                wf_IY[curr_k + i] = viy;

                // Store overall best-state direction for inner cells
                if (i > 0 && j > 0) {
                    int16_t best_val = vm;
                    uint8_t best_dir = dir;
                    if (vix > best_val) { best_val = vix; best_dir = CPU_DIR_LEFT; }
                    if (viy > best_val) {                 best_dir = CPU_DIR_UP;   }
                    tbDir[(i - 1) * T + (j - 1)] = best_dir;
                }

                // GACT overlap: track best M-state score only.
                // Using max(vm,vix,viy) would seed next tile mid-gap.
                if (!lastTile) {
                    if (i > (refLen - O) && j > (qryLen - O)) {
                        if (vm >= maxScore) {
                            maxScore = vm;
                            best_ti  = i;
                            best_tj  = j;
                        }
                    }
                }

            } // end for i (wavefront cells)

        } // end wavefront loop (k)

        // =====================================================================
        // TRACEBACK (within tile, backwards)
        // =====================================================================
        int ti = (!lastTile) ? best_ti : refLen;
        int tj = (!lastTile) ? best_tj : qryLen;

        int next_ref_advance = ti;
        int next_qry_advance = tj;
        int localLen = 0;

        while (ti > 0 || tj > 0) {
            uint8_t dir;
            if      (ti == 0) { dir = CPU_DIR_LEFT; }
            else if (tj == 0) { dir = CPU_DIR_UP;   }
            else              { dir = tbDir[(ti - 1) * T + (tj - 1)]; }

            localPath[localLen++] = dir;

            if      (dir == CPU_DIR_DIAG) { ti--; tj--; }
            else if (dir == CPU_DIR_UP)   { ti--;       }
            else                          {       tj--; }
        }

        // Write reversed path into traceback buffer (forward order)
        for (int s = localLen - 1; s >= 0; --s) {
            if (currentPathLen < tb_length)
                tb[currentPathLen++] = localPath[s];
        }

        reference_idx += next_ref_advance;
        query_idx     += next_qry_advance;

    } // end tile loop
}


// ============================================================================
// GpuAligner::allocateMem
// No device memory on CPU. Sets longestLen and nulls the CUDA pointers.
// ============================================================================
void GpuAligner::allocateMem() {
    longestLen = (int32_t)std::max_element(
        seqs.begin(), seqs.end(),
        [](const Sequence& a, const Sequence& b){
            return a.seq.size() < b.seq.size();
        }
    )->seq.size();

    d_seqs   = nullptr;
    d_tb     = nullptr;
    d_seqLen = nullptr;
    d_info   = nullptr;
}


// ============================================================================
// GpuAligner::transferSequence2Device — no-op on CPU
// ============================================================================
void GpuAligner::transferSequence2Device() {}


// ============================================================================
// GpuAligner::transferTB2Host — no-op on CPU
// The real traceback buffer is owned inside alignment() and passed directly
// to getAlignedSequences(). Returns empty vector to satisfy the signature.
// ============================================================================
TB_PATH GpuAligner::transferTB2Host() {
    return TB_PATH{};
}


// ============================================================================
// GpuAligner::getAlignedSequences
// Identical logic to the GPU version — walks the flat traceback buffer and
// reconstructs aligned strings in seqs[].
// ============================================================================
void GpuAligner::getAlignedSequences(TB_PATH& tb_paths) {

    int tb_length = longestLen << 1;

    // To parallelise: #pragma omp parallel for schedule(dynamic)
    for (int pair = 0; pair < numPairs; ++pair) {
        int tb_start = tb_length * pair;

        int seqId0 = 2 * pair;
        int seqId1 = 2 * pair + 1;
        std::string seq0 = seqs[seqId0].seq;
        std::string seq1 = seqs[seqId1].seq;
        std::string aln0, aln1;
        int seqPos0 = 0, seqPos1 = 0;

        for (int i = tb_start; i < tb_start + tb_length; ++i) {
            if (tb_paths[i] == CPU_DIR_DIAG) {
                aln0 += seq0[seqPos0];
                aln1 += seq1[seqPos1];
                seqPos0++; seqPos1++;
            }
            else if (tb_paths[i] == CPU_DIR_UP) {
                // IY state: ref advances, gap in query
                aln0 += seq0[seqPos0];
                aln1 += '-';
                seqPos0++;
            }
            else if (tb_paths[i] == CPU_DIR_LEFT) {
                // IX state: query advances, gap in ref
                aln0 += '-';
                aln1 += seq1[seqPos1];
                seqPos1++;
            }
            else {
                break;  // sentinel 0: end of path
            }
        }

        seqs[seqId0].aln = aln0;
        seqs[seqId1].aln = aln1;
    }
}


// ============================================================================
// GpuAligner::alignment
// Main orchestrator: builds flat arrays, calls alignOnePair per pair,
// then reconstructs aligned strings via getAlignedSequences.
// ============================================================================
void GpuAligner::alignment() {

    allocateMem();  // sets longestLen, nulls CUDA pointers

    int32_t tb_length = longestLen << 1;

    // Build the same flat sequence layout used by the GPU version
    std::vector<char> h_seqs((size_t)longestLen * numPairs * 2, 0);
    for (int i = 0; i < numPairs * 2; ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(h_seqs.data() + (i * longestLen), s.data(), s.size());
    }

    std::vector<int32_t> h_seqLen(numPairs * 2, 0);
    for (int i = 0; i < numPairs * 2; ++i)
        h_seqLen[i] = (int32_t)seqs[i].seq.size();

    // Traceback buffer for all pairs (zero-initialised)
    TB_PATH tb_paths((size_t)tb_length * numPairs, 0);

    // Run alignment for each pair.
    // To parallelise: #pragma omp parallel for schedule(dynamic)
    for (int pair = 0; pair < numPairs; ++pair) {
        int32_t refStart    = (pair * 2)     * longestLen;
        int32_t qryStart    = (pair * 2 + 1) * longestLen;
        int32_t refTotalLen = h_seqLen[2 * pair];
        int32_t qryTotalLen = h_seqLen[2 * pair + 1];
        int32_t tbOffset    = pair * tb_length;

        alignOnePair(
            h_seqs.data(),
            refStart,
            qryStart,
            refTotalLen,
            qryTotalLen,
            tb_paths.data() + tbOffset,
            tb_length
        );
    }

    getAlignedSequences(tb_paths);
}


// ============================================================================
// GpuAligner::clearAndReset
// ============================================================================
void GpuAligner::clearAndReset() {
    seqs.clear();
    longestLen = 0;
    numPairs   = 0;
    d_seqs   = nullptr;
    d_tb     = nullptr;
    d_seqLen = nullptr;
    d_info   = nullptr;
}


// ============================================================================
// GpuAligner::writeAlignment
// ============================================================================
void GpuAligner::writeAlignment(std::string fileName, bool append) {
    std::ofstream outFile;
    if (append) outFile.open(fileName, std::ios::app);
    else        outFile.open(fileName);
    if (!outFile) {
        fprintf(stderr, "ERROR: cannot open file: %s\n", fileName.c_str());
        exit(1);
    }
    for (auto& seq : seqs) {
        outFile << ('>' + seq.name + '\n');
        outFile << (seq.aln  + '\n');
    }
    outFile.close();
}


// ============================================================================
// printGpuProperties — stub required by the header, no GPU on CPU build
// ============================================================================
void printGpuProperties() {
    printf("CPU build — no GPU available.\n");
}
