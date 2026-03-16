#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <math.h>

// =============================================================================
// CUDA error checking macro
//
// Wraps any CUDA API call. On failure, prints the file, line, and error string
// to stderr and exits. Usage: CUDA_CHECK(cudaMalloc(...));
// =============================================================================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d - %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while(0)

// =============================================================================
// allocateMem
//
// Allocates all GPU device memory needed for one batch of sequence pairs.
//
// Key difference from the single-thread version:
//   d_tbDir and d_tbState are now sized numPairs * 3 * T_TILE * T_TILE.
//   The single-thread version only needed one shared scratch tile because
//   pairs were processed one-at-a-time serially. Here, all numPairs blocks
//   run simultaneously, so each block needs its own private scratch region
//   to avoid writes from one pair overwriting another pair's traceback data.
//   Block p accesses: d_tbDir + p * 3 * T_TILE * T_TILE
// =============================================================================
void GpuAligner::allocateMem() {
    // Determine the longest sequence — this sets the uniform stride for d_seqs
    longestLen = std::max_element(seqs.begin(), seqs.end(),
        [](const Sequence& a, const Sequence& b) {
            return a.seq.size() < b.seq.size();
        })->seq.size();

    cudaError_t err;

    // Flat packed sequence array: all references and queries, each padded to longestLen
    err = cudaMalloc(&d_seqs, numPairs * 2 * longestLen * sizeof(char));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Actual (unpadded) sequence lengths — 2 entries per pair (ref, qry)
    err = cudaMalloc(&d_seqLen, numPairs * 2 * sizeof(int32_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Output traceback buffer: worst-case 2*longestLen direction codes per pair
    int tb_length = longestLen << 1;
    err = cudaMalloc(&d_tb, numPairs * tb_length * sizeof(uint8_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Two-element info array passed to the kernel: [numPairs, longestLen]
    err = cudaMalloc(&d_info, 2 * sizeof(int32_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // CHANGE: d_tbDir/d_tbState now sized numPairs * 3 * T * T so each block
    // (pair) has its own private scratch region — no inter-pair interference.
    err = cudaMalloc(&d_tbDir,   (size_t)numPairs * 3 * T_TILE * T_TILE * sizeof(uint8_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    err = cudaMalloc(&d_tbState, (size_t)numPairs * 3 * T_TILE * T_TILE * sizeof(uint8_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }
}

// =============================================================================
// transferSequence2Device
//
// Packs the host Sequence objects into flat device arrays and copies them to
// GPU memory. Also zero-initialises d_tb on the device so unwritten path
// entries read as 0 (the end-of-path sentinel for getAlignedSequences).
// =============================================================================
void GpuAligner::transferSequence2Device() {
    cudaError_t err;

    // Build the flat host sequence array: each sequence zero-padded to longestLen
    std::vector<char> h_seqs(longestLen * numPairs * 2, 0);
    for (size_t i = 0; i < (size_t)(numPairs * 2); ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(h_seqs.data() + (i * longestLen), s.data(), s.size());
    }
    err = cudaMemcpy(d_seqs, h_seqs.data(),
                     longestLen * numPairs * 2 * sizeof(char),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Copy actual sequence lengths so the kernel knows where valid data ends
    std::vector<int32_t> h_seqLen(numPairs * 2, 0);
    for (int i = 0; i < numPairs * 2; ++i) h_seqLen[i] = seqs[i].seq.size();
    err = cudaMemcpy(d_seqLen, h_seqLen.data(),
                     numPairs * 2 * sizeof(int32_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Zero-initialise the traceback buffer — 0 is the end-of-path sentinel
    int tb_length = longestLen << 1;
    std::vector<uint8_t> h_tb(tb_length * numPairs, 0);
    err = cudaMemcpy(d_tb, h_tb.data(),
                     tb_length * numPairs * sizeof(uint8_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Copy the two-element info array: [numPairs, longestLen]
    std::vector<int32_t> h_info(2);
    h_info[0] = numPairs;
    h_info[1] = longestLen;
    err = cudaMemcpy(d_info, h_info.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }
}

// =============================================================================
// transferTB2Host
//
// Copies the completed traceback path buffer from GPU global memory to a host
// std::vector for use by getAlignedSequences().
// =============================================================================
TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen << 1;
    TB_PATH h_tb(tb_length * numPairs);
    cudaError_t err = cudaMemcpy(h_tb.data(), d_tb,
                                 tb_length * numPairs * sizeof(uint8_t),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }
    return h_tb;
}

// =============================================================================
// alignmentOnGPU — Level 1: one block per pair (inter-pair parallelism)
//
// WHAT CHANGED FROM THE SINGLE-THREAD VERSION (Level 0)
// ------------------------------------------------------
// Level 0: 1 block, 1 thread. An outer for-loop iterated over all pairs one
//          by one. All pairs ran serially on a single thread.
//
// Level 1: numPairs blocks, 1 thread each. The outer for-loop is gone.
//          Instead, pair = blockIdx.x. Every block handles exactly one pair,
//          and all blocks execute simultaneously on the GPU's SMs.
//          This is Level 1 (inter-pair) parallelism: all pairs run in parallel.
//
// The HMM scoring logic, GACT tiling, wavefront traversal, traceback, and
// path write are all identical to the single-thread version. The only
// structural change is replacing the for-loop with blockIdx.x.
//
// TRACEBACK BUFFER STRIDING
// -------------------------
// In the single-thread version, d_tbDir/d_tbState were a single 3*T*T scratch
// region reused for every tile of every pair. Now that all pairs run at the
// same time, each block must have its own private region:
//   block p → tbDir   = d_tbDir   + p * 3 * T_TILE * T_TILE
//   block p → tbState = d_tbState + p * 3 * T_TILE * T_TILE
// This prevents blocks from overwriting each other's traceback data mid-tile.
//
// SHARED MEMORY USAGE
// -------------------
// The wavefront arrays (wf_M, wf_I, wf_D) and localPath remain in shared
// memory — they are private per block by definition (shared memory is not
// visible between blocks), so no additional changes are needed there.
//
// Per-tile scalar variables (lastTile, carryLogProb, best_ti, etc.) no longer
// need __shared__ because there is only one thread per block. They become
// ordinary thread-local variables.
// =============================================================================

// HMM log-transition probabilities — identical values to the single-thread version.
// Double precision used throughout to match the CPU reference exactly.
#define LOG_MM            -0.10536051565782630   // log(0.90)  M→M
#define LOG_MI            -2.99573227355399099   // log(0.05)  M→I
#define LOG_MD            -2.99573227355399099   // log(0.05)  M→D
#define LOG_IM            -0.35667494393873245   // log(0.70)  I→M
#define LOG_II            -1.20397280432593597   // log(0.30)  I→I
#define LOG_DM            -0.35667494393873245   // log(0.70)  D→M
#define LOG_DD            -1.20397280432593597   // log(0.30)  D→D
#define LOG_INIT_M        -0.10536051565782630   // log(0.90)  initial probability of state M
#define LOG_INIT_I        -2.99573227355399099   // log(0.05)  initial probability of state I
#define LOG_INIT_D        -2.99573227355399099   // log(0.05)  initial probability of state D
#define LOG_EMIT_MATCH    -0.10536051565782630   // log(0.90)  emission: ref[i] == qry[j]
#define LOG_EMIT_MISMATCH -2.30258509299404568   // log(0.10)  emission: ref[i] != qry[j]
#define NEG_INF_D         -1e18                  // log(0) — unreachable state sentinel

// T_TILE is defined in alignment.cuh (= 200)
// O_TILE: size of the overlap corner scanned at the end of each non-final tile
#define O_TILE 64

__global__ void alignmentOnGPU(
    int32_t* d_info,       // [0]=numPairs, [1]=longestLen (sequence stride)
    int32_t* d_seqLen,     // actual length of each sequence (2*numPairs entries)
    char*    d_seqs,       // flat packed sequences, stride=longestLen
    uint8_t* d_tb,         // output: traceback direction codes, one path per pair
    uint8_t* d_tbDir,      // global scratch: numPairs * 3 * T_TILE * T_TILE
    uint8_t* d_tbState)    // global scratch: numPairs * 3 * T_TILE * T_TILE
{
    // CHANGE: pair index comes from blockIdx.x, not an inner for-loop.
    // tx==0 guard kept: still 1 thread per block (Step 1 only).
    const int pair = blockIdx.x;
    const int tx   = threadIdx.x;
    if (tx != 0) return;  // only thread 0 of each block does work (blockSize=1 anyway)

    const uint8_t DIR_DIAG = 1;   // match/mismatch: both sequences advance
    const uint8_t DIR_UP   = 2;   // deletion: only ref advances (gap in query)
    const uint8_t DIR_LEFT = 3;   // insertion: only query advances (gap in ref)
    const int S_M = 0, S_I = 1, S_D = 2;
    const int CELL = T_TILE * T_TILE;  // stride between state planes in tbDir/tbState

    const int32_t maxSeqLen = d_info[1];  // = longestLen, stride between sequences in d_seqs

    // Guard: skip if blockIdx.x is out of range (in case launch rounds up).
    if (pair >= d_info[0]) return;

    // CHANGE: d_tbDir/d_tbState base pointer is offset by pair so each block
    // writes into its own private region, no conflicts between blocks.
    uint8_t* tbDir   = d_tbDir   + (size_t)pair * 3 * CELL;
    uint8_t* tbState = d_tbState + (size_t)pair * 3 * CELL;

    // Wavefront and path buffers — still in shared memory (private per block).
    // wf_M/I/D: cyclic triple-buffer holding one diagonal's scores per state.
    // localPath: temporary storage for the reversed traceback path of one tile.
    __shared__ double  wf_M[3 * (T_TILE + 1)];
    __shared__ double  wf_I[3 * (T_TILE + 1)];
    __shared__ double  wf_D[3 * (T_TILE + 1)];
    __shared__ uint8_t localPath[2 * T_TILE];

    // Per-tile scalars — no longer need __shared__ since only 1 thread per block.
    // In Level 2+, these become __shared__ again so all threads in the block can
    // read them without explicit broadcasts.
    bool    lastTile     = false;       // true when current tile covers end of both sequences
    double  carryLogProb = 0.0;         // log(1) neutral seed for the first tile's origin
    int32_t best_ti, best_tj, best_state;  // best overlap cell found in this tile

    int32_t currentPairPathLen = 0;  // number of direction codes written so far for this pair
    int32_t reference_idx      = 0;  // tile start offset into the reference sequence
    int32_t query_idx          = 0;  // tile start offset into the query sequence

    // Byte offsets into d_seqs and d_tb for this pair
    const int32_t refStart       = (pair * 2)     * maxSeqLen;  // reference sequence start
    const int32_t qryStart       = (pair * 2 + 1) * maxSeqLen;  // query sequence start
    const int32_t tbGlobalOffset = pair * (maxSeqLen * 2);       // traceback path start in d_tb

    const int32_t refTotalLen = d_seqLen[2 * pair];      // full reference length
    const int32_t qryTotalLen = d_seqLen[2 * pair + 1];  // full query length

    // -----------------------------------------------------------------------
    // TILE LOOP — unchanged logic, just runs in its own block now
    // -----------------------------------------------------------------------
    while (!lastTile) {

        // Clamp tile dimensions to remaining sequence lengths
        int32_t refLen = min(T_TILE, (int)(refTotalLen - reference_idx));
        int32_t qryLen = min(T_TILE, (int)(qryTotalLen - query_idx));

        // Mark as last tile when this tile reaches the end of both sequences
        if ((reference_idx + refLen == refTotalLen) &&
            (query_idx     + qryLen == qryTotalLen))
            lastTile = true;

        // Init wavefront buffers — reset all three diagonal slots to NEG_INF_D
        for (int s = 0; s < 3 * (T_TILE + 1); ++s) {
            wf_M[s] = NEG_INF_D;
            wf_I[s] = NEG_INF_D;
            wf_D[s] = NEG_INF_D;
        }

        // Reset overlap tracking — default to terminal cell in case nothing beats NEG_INF_D
        double bestOverlapScore = NEG_INF_D;
        best_ti    = refLen;
        best_tj    = qryLen;
        best_state = S_M;

        // -------------------------------------------------------------------
        // WAVEFRONT (ANTI-DIAGONAL) LOOP — identical to single-thread version
        //
        // Iterates anti-diagonal k = i+j from 0 to refLen+qryLen.
        // All cells (i,j) on the same anti-diagonal are independent, which
        // is the basis for Level 2's thread-level parallelism. Here in Level 1
        // they are still computed serially by the single thread.
        //
        // Cyclic triple-buffer: curr_k writes diagonal k, pre_k holds k-1,
        // prepre_k holds k-2. States I and D read from k-1; state M reads k-2.
        // -------------------------------------------------------------------
        for (int k = 0; k <= refLen + qryLen; ++k) {

            int curr_k   = (k     % 3) * (T_TILE + 1);  // write slot for diagonal k
            int pre_k    = ((k+2) % 3) * (T_TILE + 1);  // diagonal k-1
            int prepre_k = ((k+1) % 3) * (T_TILE + 1);  // diagonal k-2

            // i ranges over valid ref indices on this anti-diagonal
            // (j = k-i must stay within [0, qryLen])
            int i_start = max(0,           k - (int)qryLen);
            int i_end   = min((int)refLen, k);

            for (int i = i_start; i <= i_end; ++i) {
                int j = k - i;

                // Local scores and predecessor states for this cell
                double  vm = NEG_INF_D, vi = NEG_INF_D, vd = NEG_INF_D;
                uint8_t pre_m = (uint8_t)S_M;
                uint8_t pre_i = (uint8_t)S_M;
                uint8_t pre_d = (uint8_t)S_M;

                if (i == 0 && j == 0) {
                    // Tile origin — seed all three states from carry-in score
                    // V_s[0][0] = carryLogProb + log π(s)
                    vm = carryLogProb + LOG_INIT_M;
                    vi = carryLogProb + LOG_INIT_I;
                    vd = carryLogProb + LOG_INIT_D;
                }
                else if (i == 0) {
                    // Top boundary (i=0, j>0): only query gaps are possible.
                    // Only state I is active. V_I[0][j] = max(V_M[0][j-1]+logT[M→I], V_I[0][j-1]+logT[I→I])
                    double fMI = wf_M[pre_k + 0] + LOG_MI;
                    double fII = wf_I[pre_k + 0] + LOG_II;
                    if (fMI >= fII) { vi = fMI; pre_i = (uint8_t)S_M; }
                    else            { vi = fII; pre_i = (uint8_t)S_I; }
                    tbDir  [S_I * CELL + 0 * T_TILE + (j-1)] = DIR_LEFT;
                    tbState[S_I * CELL + 0 * T_TILE + (j-1)] = pre_i;
                }
                else if (j == 0) {
                    // Left boundary (i>0, j=0): only ref gaps are possible.
                    // Only state D is active. V_D[i][0] = max(V_M[i-1][0]+logT[M→D], V_D[i-1][0]+logT[D→D])
                    double fMD = wf_M[pre_k + (i-1)] + LOG_MD;
                    double fDD = wf_D[pre_k + (i-1)] + LOG_DD;
                    if (fMD >= fDD) { vd = fMD; pre_d = (uint8_t)S_M; }
                    else            { vd = fDD; pre_d = (uint8_t)S_D; }
                    tbDir  [S_D * CELL + (i-1) * T_TILE + 0] = DIR_UP;
                    tbState[S_D * CELL + (i-1) * T_TILE + 0] = pre_d;
                }
                else {
                    // Inner cell: compute all three HMM states
                    char   r    = d_seqs[refStart + reference_idx + (i-1)];  // ref base
                    char   q    = d_seqs[qryStart + query_idx     + (j-1)];  // qry base
                    double emit = (r == q) ? LOG_EMIT_MATCH : LOG_EMIT_MISMATCH;

                    // State M — diagonal predecessor at (i-1,j-1), diagonal k-2
                    // All three predecessor states allowed. V_M[i][j] = emit + max_s(V_s[i-1][j-1]+logT[s→M])
                    double mMM = wf_M[prepre_k + (i-1)] + LOG_MM;
                    double mIM = wf_I[prepre_k + (i-1)] + LOG_IM;
                    double mDM = wf_D[prepre_k + (i-1)] + LOG_DM;
                    if (mMM >= mIM && mMM >= mDM) { vm = mMM + emit; pre_m = S_M; }
                    else if (mIM >= mDM)           { vm = mIM + emit; pre_m = S_I; }
                    else                           { vm = mDM + emit; pre_m = S_D; }

                    // State I — leftward predecessor at (i,j-1), diagonal k-1. D→I forbidden.
                    // V_I[i][j] = max(V_M[i][j-1]+logT[M→I], V_I[i][j-1]+logT[I→I])
                    double iMI = wf_M[pre_k + i] + LOG_MI;
                    double iII = wf_I[pre_k + i] + LOG_II;
                    if (iMI >= iII) { vi = iMI; pre_i = S_M; }
                    else            { vi = iII; pre_i = S_I; }

                    // State D — upward predecessor at (i-1,j), diagonal k-1. I→D forbidden.
                    // V_D[i][j] = max(V_M[i-1][j]+logT[M→D], V_D[i-1][j]+logT[D→D])
                    double dMD = wf_M[pre_k + (i-1)] + LOG_MD;
                    double dDD = wf_D[pre_k + (i-1)] + LOG_DD;
                    if (dMD >= dDD) { vd = dMD; pre_d = S_M; }
                    else            { vd = dDD; pre_d = S_D; }

                    // Store traceback entries for all three states at this inner cell.
                    // Layout: tbDir[s * CELL + (i-1)*T_TILE + (j-1)]
                    int cell = (i-1) * T_TILE + (j-1);
                    tbDir  [S_M * CELL + cell] = DIR_DIAG; tbState[S_M * CELL + cell] = pre_m;
                    tbDir  [S_I * CELL + cell] = DIR_LEFT; tbState[S_I * CELL + cell] = pre_i;
                    tbDir  [S_D * CELL + cell] = DIR_UP;   tbState[S_D * CELL + cell] = pre_d;
                }

                // Commit scores to the current wavefront slot
                wf_M[curr_k + i] = vm;
                wf_I[curr_k + i] = vi;
                wf_D[curr_k + i] = vd;

                // GACT overlap tracking: scan the bottom-right O_TILE×O_TILE corner.
                // On equal scores, last state wins (>= with s=0,1,2 → S_D beats S_I beats S_M).
                // This tie-breaking must be preserved exactly in Level 2's parallel reduction.
                if (!lastTile &&
                    i > (refLen - O_TILE) && j > (qryLen - O_TILE)) {
                    double scores[3] = {vm, vi, vd};
                    for (int s = 0; s < 3; ++s) {
                        if (scores[s] >= bestOverlapScore) {
                            bestOverlapScore = scores[s];
                            best_state       = s;
                            best_ti          = i;
                            best_tj          = j;
                        }
                    }
                }
            }
        } // end wavefront loop

        // -------------------------------------------------------------------
        // TRACEBACK — identical logic to single-thread version
        //
        // Walk backwards from the best overlap cell (non-final tile) or the
        // terminal cell (final tile) to (0,0), following the stored directions
        // and predecessor states. cur_state selects the correct table row.
        // Directions are collected in reverse into localPath[], then reversed
        // into the global d_tb buffer.
        // -------------------------------------------------------------------
        int ti = (!lastTile) ? (int)best_ti : (int)refLen;  // traceback start (ref axis)
        int tj = (!lastTile) ? (int)best_tj : (int)qryLen;  // traceback start (qry axis)

        int next_ref_advance = ti;  // how far to advance reference_idx after this tile
        int next_qry_advance = tj;  // how far to advance query_idx after this tile

        // Update carry-in score for the next tile's origin cell (0,0)
        carryLogProb = (!lastTile) ? bestOverlapScore : NEG_INF_D;
        if (lastTile) {
            // Final tile: read the terminal scores from the wavefront buffer
            int curr_k = (((int)refLen + (int)qryLen) % 3) * (T_TILE + 1);
            double fM = wf_M[curr_k + refLen];
            double fI = wf_I[curr_k + refLen];
            double fD = wf_D[curr_k + refLen];
            carryLogProb = fM;
            if (fI > carryLogProb) carryLogProb = fI;
            if (fD > carryLogProb) carryLogProb = fD;
        }

        int localLen = 0;

        // Select the starting HMM state for traceback
        int cur_state;
        if (!lastTile) {
            cur_state = (int)best_state;  // state at the best overlap cell
        } else {
            // Final tile: argmax over the three terminal state scores
            int curr_k = (((int)refLen + (int)qryLen) % 3) * (T_TILE + 1);
            double fM = wf_M[curr_k + refLen];
            double fI = wf_I[curr_k + refLen];
            double fD = wf_D[curr_k + refLen];
            cur_state = S_M;
            if (fI > fM && fI > fD) cur_state = S_I;
            else if (fD > fM)        cur_state = S_D;
        }

        // Walk backwards: cur_state selects which TB table row to read at each step.
        // TB_dir gives the move direction; TB_state gives the predecessor state.
        while (ti > 0 || tj > 0) {
            uint8_t dir;
            int     prev_state;

            if (ti == 0) {
                // Top boundary — forced DIR_LEFT (query gap), must be in state I
                dir        = DIR_LEFT;
                prev_state = (int)tbState[S_I * CELL + 0 * T_TILE + (tj-1)];
                tj--;
            } else if (tj == 0) {
                // Left boundary — forced DIR_UP (ref gap), must be in state D
                dir        = DIR_UP;
                prev_state = (int)tbState[S_D * CELL + (ti-1) * T_TILE + 0];
                ti--;
            } else {
                // Inner cell — direction and predecessor determined by cur_state
                int cell   = (ti-1) * T_TILE + (tj-1);
                dir        = tbDir  [cur_state * CELL + cell];
                prev_state = tbState[cur_state * CELL + cell];
                if      (dir == DIR_DIAG) { ti--; tj--; }
                else if (dir == DIR_UP)   { ti--;        }
                else                      {       tj--;  }
            }

            localPath[localLen++] = dir;  // collected in reverse order
            cur_state = prev_state;        // step to the predecessor state
        }

        // Reverse localPath into the global traceback buffer (paths stored forward)
        for (int k = localLen - 1; k >= 0; --k) {
            d_tb[tbGlobalOffset + currentPairPathLen] = localPath[k];
            currentPairPathLen++;
        }

        // Advance sequence cursors for the next tile
        reference_idx += next_ref_advance;
        query_idx     += next_qry_advance;

    } // end tile loop
}

// =============================================================================
// getAlignedSequences
//
// Reconstructs gapped alignment strings from the compact traceback path.
//   DIR_DIAG (1): consume one base from each sequence (match or mismatch)
//   DIR_UP   (2): consume one ref base, insert '-' into query alignment
//   DIR_LEFT (3): insert '-' into ref alignment, consume one query base
//   0 / other:    end-of-path sentinel — stop
// =============================================================================
void GpuAligner::getAlignedSequences(TB_PATH& tb_paths) {
    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    int tb_length = longestLen << 1;  // allocated path length per pair

    for (int pair = 0; pair < numPairs; ++pair) {
        int tb_start = tb_length * pair;  // offset into tb_paths for this pair
        int seqId0   = 2 * pair;
        int seqId1   = 2 * pair + 1;
        std::string seq0 = seqs[seqId0].seq;
        std::string seq1 = seqs[seqId1].seq;
        std::string aln0, aln1;
        int seqPos0 = 0, seqPos1 = 0;

        for (int i = tb_start; i < tb_start + tb_length; ++i) {
            if      (tb_paths[i] == DIR_DIAG) { aln0 += seq0[seqPos0++]; aln1 += seq1[seqPos1++]; }
            else if (tb_paths[i] == DIR_UP)   { aln0 += seq0[seqPos0++]; aln1 += '-'; }
            else if (tb_paths[i] == DIR_LEFT) { aln0 += '-';             aln1 += seq1[seqPos1++]; }
            else break;  // 0 = end-of-path sentinel
        }

        seqs[seqId0].aln = aln0;
        seqs[seqId1].aln = aln1;
    }
}

// =============================================================================
// clearAndReset
//
// Frees all GPU device memory and resets the aligner state so the same
// GpuAligner object can be reused for the next batch.
// =============================================================================
void GpuAligner::clearAndReset() {
    cudaFree(d_seqs);
    cudaFree(d_seqLen);
    cudaFree(d_tb);
    cudaFree(d_info);
    cudaFree(d_tbDir);
    cudaFree(d_tbState);
    seqs.clear();
    longestLen = 0;
    numPairs   = 0;
}

// =============================================================================
// alignment
//
// Orchestrates the Level 1 GPU alignment:
//   1. Allocate GPU memory
//   2. Transfer sequences and metadata to the device
//   3. Launch numPairs blocks × 1 thread — each block handles one pair
//   4. Synchronise, transfer traceback paths back to host
//   5. Reconstruct aligned strings
// =============================================================================
void GpuAligner::alignment() {
    // CHANGE: launch numPairs blocks, 1 thread each.
    // Each block handles exactly one pair independently.
    const int numBlocks = numPairs;
    const int blockSize = 1;

    allocateMem();
    transferSequence2Device();

    alignmentOnGPU<<<numBlocks, blockSize>>>(d_info, d_seqLen, d_seqs, d_tb,
                                             d_tbDir, d_tbState);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();
    TB_PATH tb_paths = transferTB2Host();
    getAlignedSequences(tb_paths);
}

// =============================================================================
// writeAlignment
//
// Writes all aligned sequences to a FASTA file.
// append=true opens in append mode (for multi-batch output to one file).
// =============================================================================
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
