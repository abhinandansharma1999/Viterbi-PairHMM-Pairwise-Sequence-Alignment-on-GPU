#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <math.h>

// =============================================================================
// CUDA error checking macro
//
// Wraps any CUDA API call. If the call fails, prints the file name, line
// number, and human-readable error string to stderr, then exits.
// Usage: CUDA_CHECK(cudaMalloc(...));
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
// Allocates all GPU (device) memory needed for one batch of sequence pairs.
// Must be called before transferSequence2Device() and the kernel launch.
//
// Device memory layout:
//   d_seqs   : flat array of all sequences, each zero-padded to longestLen bytes.
//              Sequence i lives at d_seqs + i * longestLen.
//              Even indices are references, odd indices are queries:
//              pair p → ref = d_seqs[2p * longestLen], qry = d_seqs[(2p+1) * longestLen]
//
//   d_seqLen : actual (unpadded) length of each sequence. Length of seq i = d_seqLen[i].
//
//   d_tb     : output traceback buffer. Each pair gets 2*longestLen bytes
//              (worst case: all gaps). Pair p starts at d_tb + p * 2 * longestLen.
//
//   d_info   : two-element array: d_info[0]=numPairs, d_info[1]=longestLen (stride).
//
//   d_tbDir  : per-tile, per-state traceback direction table.
//              d_tbDir[s * T*T + (i-1)*T + (j-1)] = direction that reached
//              state s at tile cell (i,j). Size = 3 * T_TILE * T_TILE bytes.
//              Stored in global memory because 3*200*200 = 120 KB exceeds the
//              48 KB shared memory limit per block.
//
//   d_tbState: per-tile, per-state traceback predecessor table.
//              d_tbState[s * T*T + (i-1)*T + (j-1)] = which HMM state fed
//              into state s at tile cell (i,j). Same size and layout as d_tbDir.
//              Together with d_tbDir, these two tables enable state-aware
//              traceback: at each step, cur_state selects the correct row.
// =============================================================================
void GpuAligner::allocateMem() {
    // Find the longest sequence to determine the uniform stride for d_seqs
    longestLen = std::max_element(seqs.begin(), seqs.end(),
        [](const Sequence& a, const Sequence& b) {
            return a.seq.size() < b.seq.size();
        })->seq.size();

    cudaError_t err;

    // Flat packed sequence array: numPairs pairs × 2 sequences × longestLen bytes
    err = cudaMalloc(&d_seqs, numPairs * 2 * longestLen * sizeof(char));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Actual lengths — the kernel uses these to avoid processing padding zeros
    err = cudaMalloc(&d_seqLen, numPairs * 2 * sizeof(int32_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Output traceback path: worst-case 2*longestLen direction codes per pair
    int tb_length = longestLen << 1;
    err = cudaMalloc(&d_tb, numPairs * tb_length * sizeof(uint8_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Two-element info array passed to the kernel: [numPairs, longestLen]
    err = cudaMalloc(&d_info, 2 * sizeof(int32_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Tile-scratch buffers for per-state traceback tables (3 states x T x T).
    // 3*200*200 = 120 KB each — far over the 48 KB shared memory limit, so
    // kept in global device memory and reused every tile.
    err = cudaMalloc(&d_tbDir,   3 * T_TILE * T_TILE * sizeof(uint8_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    err = cudaMalloc(&d_tbState, 3 * T_TILE * T_TILE * sizeof(uint8_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }
}

// =============================================================================
// transferSequence2Device
//
// Packs the host Sequence objects into the flat device arrays and copies them
// to GPU memory. Also zero-initialises the traceback buffer d_tb on the device
// so unwritten entries read as 0 (the end-of-path sentinel).
// =============================================================================
void GpuAligner::transferSequence2Device() {
    cudaError_t err;

    // Build the flat host-side sequence array, zero-padded to longestLen per sequence
    std::vector<char> h_seqs(longestLen * numPairs * 2, 0);
    for (size_t i = 0; i < (size_t)(numPairs * 2); ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(h_seqs.data() + (i * longestLen), s.data(), s.size());
    }
    err = cudaMemcpy(d_seqs, h_seqs.data(),
                     longestLen * numPairs * 2 * sizeof(char),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Copy actual (unpadded) sequence lengths to device
    std::vector<int32_t> h_seqLen(numPairs * 2, 0);
    for (int i = 0; i < numPairs * 2; ++i) h_seqLen[i] = seqs[i].seq.size();
    err = cudaMemcpy(d_seqLen, h_seqLen.data(),
                     numPairs * 2 * sizeof(int32_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Zero-initialise traceback buffer — 0 is the end-of-path sentinel read by getAlignedSequences
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
// Copies the completed traceback path buffer from GPU global memory back to
// a host std::vector for consumption by getAlignedSequences().
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
// alignmentOnGPU — Single-thread GPU kernel, Viterbi Pair-HMM with GACT tiling
//
// PURPOSE
// -------
// This kernel is the GPU baseline: it runs on exactly ONE thread (block 0,
// thread 0) and processes all pairs sequentially in a for-loop. Its output
// is bit-for-bit identical to the CPU GACT implementation, making it the
// correctness reference for all subsequent parallel GPU versions.
// The single-thread design eliminates any race conditions or reduction errors,
// so any divergence from the CPU output is immediately attributable to a
// logic bug rather than a parallelism bug.
//
// HMM STATE MODEL
// ---------------
//   M=0  Match/Mismatch  — both sequences advance  → DIR_DIAG  (diagonal move)
//   I=1  Query gap       — only query advances     → DIR_LEFT  (leftward move)
//   D=2  Reference gap   — only ref advances       → DIR_UP    (upward move)
//
// TRACEBACK TABLE LAYOUT
// ----------------------
// The traceback tables are indexed as [state][i-1][j-1] (flattened):
//   d_tbDir  [s * CELL + (i-1)*T_TILE + (j-1)] = direction for state s at (i,j)
//   d_tbState[s * CELL + (i-1)*T_TILE + (j-1)] = predecessor state for state s at (i,j)
//
// Three separate traceback entries per cell are necessary because the same
// cell (i,j) can be reached via different predecessor states depending on
// which HMM state we are tracing. cur_state selects the correct row.
//
// WAVEFRONT (ANTI-DIAGONAL) TRAVERSAL
// ------------------------------------
// Cells are visited along anti-diagonals k = i+j (0 ≤ k ≤ refLen+qryLen).
// A cyclic triple-buffer (indices curr_k, pre_k, prepre_k) holds the scores
// for the current diagonal and the two preceding ones — the only dependencies
// needed for the three HMM state recurrences:
//   V_M reads from diagonal k-2  (predecessor was at i-1, j-1)
//   V_I reads from diagonal k-1  (predecessor was at i,   j-1)
//   V_D reads from diagonal k-1  (predecessor was at i-1, j  )
//
// GACT TILING
// -----------
// Long sequences are split into T_TILE × T_TILE tiles processed sequentially.
// At the end of each non-final tile, the best-scoring cell in the bottom-right
// O_TILE × O_TILE overlap corner is identified. Its score (carryLogProb) and
// state (best_state) seed the next tile's origin cell (0,0), so the alignment
// path can drift relative to the main diagonal between tiles.
// =============================================================================

// HMM log-transition probabilities — stored as compile-time constants.
// All scoring uses double precision to match the CPU reference exactly.
// See alignment_CPU_execution.cpp for the probability values they derive from.
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
#define NEG_INF_D         -1e18                  // represents log(0) — unreachable state

// T_TILE is defined in alignment.cuh (= 200)
// O_TILE: size of the overlap corner scanned at the end of each non-final tile
#define O_TILE   64

__global__ void alignmentOnGPU(
    int32_t* d_info,       // [0]=numPairs, [1]=longestLen (sequence stride)
    int32_t* d_seqLen,     // actual length of each sequence (2*numPairs entries)
    char*    d_seqs,       // flat packed sequences, stride=longestLen
    uint8_t* d_tb,         // output: traceback direction codes, one path per pair
    uint8_t* d_tbDir,      // tile scratch: traceback directions,   3 * T_TILE * T_TILE
    uint8_t* d_tbState)    // tile scratch: traceback predecessors, 3 * T_TILE * T_TILE
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // Only a single thread executes the entire kernel — this is the baseline.
    // All parallelism is absent here by design; correctness is the sole goal.
    if (bx == 0 && tx == 0) {

        const uint8_t DIR_DIAG = 1;   // both sequences advance (match/mismatch)
        const uint8_t DIR_UP   = 2;   // only ref advances (gap in query)
        const uint8_t DIR_LEFT = 3;   // only query advances (gap in ref)
        const int S_M = 0, S_I = 1, S_D = 2;
        const int CELL = T_TILE * T_TILE;   // stride between state planes in d_tbDir/d_tbState

        int32_t numPairs  = d_info[0];
        int32_t maxSeqLen = d_info[1];  // = longestLen, the stride between sequences in d_seqs

        // Shared memory wavefront buffers — cyclic triple-buffer, one per HMM state.
        // Each buffer holds T_TILE+1 scores (index 0 is the boundary row/column).
        // Only ~14.8 KB total, well within the 48 KB shared memory limit.
        //   wf_M/I/D : 3 * (T_TILE+1) * 8 bytes = ~4.8 KB each  → ~14.4 KB total
        //   localPath: 2 * T_TILE bytes           =   400 B
        //   scalars  : ~48 B
        __shared__ double  wf_M[3 * (T_TILE + 1)];  // Viterbi scores for state M, 3 diagonals
        __shared__ double  wf_I[3 * (T_TILE + 1)];  // Viterbi scores for state I, 3 diagonals
        __shared__ double  wf_D[3 * (T_TILE + 1)];  // Viterbi scores for state D, 3 diagonals
        __shared__ uint8_t localPath[2 * T_TILE];    // temporary traceback path (reversed), then written forward

        // Tile-level state shared between the wavefront fill and the traceback
        __shared__ bool    lastTile;      // true once the tile covers the end of both sequences
        __shared__ double  carryLogProb;  // best overlap score carried into the next tile's origin
        __shared__ int32_t best_ti;       // ref-index of the best overlap cell (next tile seed)
        __shared__ int32_t best_tj;       // qry-index of the best overlap cell (next tile seed)
        __shared__ int32_t best_state;    // HMM state at the best overlap cell

        // -----------------------------------------------------------------------
        // PAIR LOOP — process each sequence pair sequentially
        // -----------------------------------------------------------------------
        for (int pair = 0; pair < numPairs; ++pair) {

            lastTile     = false;
            carryLogProb = 0.0;   // log(1) neutral seed — first tile starts with probability 1
            int32_t currentPairPathLen = 0;  // number of direction codes written for this pair so far
            int32_t reference_idx      = 0;  // current start position in the reference (tile offset)
            int32_t query_idx          = 0;  // current start position in the query (tile offset)

            // Byte offsets into d_seqs and d_tb for this pair
            int32_t refStart       = (pair * 2)     * maxSeqLen;  // start of reference sequence
            int32_t qryStart       = (pair * 2 + 1) * maxSeqLen;  // start of query sequence
            int32_t tbGlobalOffset = pair * (maxSeqLen * 2);       // start of this pair's path in d_tb

            int32_t refTotalLen = d_seqLen[2 * pair];      // full reference length
            int32_t qryTotalLen = d_seqLen[2 * pair + 1];  // full query length

            // -------------------------------------------------------------------
            // TILE LOOP — iterate over T_TILE×T_TILE tiles along the alignment path
            // -------------------------------------------------------------------
            while (!lastTile) {

                // Clamp tile dimensions to the remaining sequence lengths
                int32_t refLen = min(T_TILE, (int)(refTotalLen - reference_idx));
                int32_t qryLen = min(T_TILE, (int)(qryTotalLen - query_idx));

                // Mark as last tile when both sequences are fully consumed
                if ((reference_idx + refLen == refTotalLen) &&
                    (query_idx     + qryLen == qryTotalLen))
                    lastTile = true;

                // Initialise all three wavefront buffers to NEG_INF_D (log 0).
                // The cyclic buffer has 3 slots of (T_TILE+1) each — indices
                // 0..T_TILE+0 for slot 0, T_TILE+1..2T_TILE+1 for slot 1, etc.
                for (int s = 0; s < 3 * (T_TILE + 1); ++s) {
                    wf_M[s] = NEG_INF_D;
                    wf_I[s] = NEG_INF_D;
                    wf_D[s] = NEG_INF_D;
                }

                // Reset overlap tracking for this tile.
                // Defaults to the terminal cell (refLen, qryLen) in case no overlap
                // cell beats NEG_INF_D (only happens if the tile is entirely unreachable).
                double bestOverlapScore = NEG_INF_D;
                best_ti    = refLen;
                best_tj    = qryLen;
                best_state = S_M;

                // ---------------------------------------------------------------
                // WAVEFRONT (ANTI-DIAGONAL) LOOP
                //
                // Anti-diagonal k covers all cells (i,j) where i+j = k.
                // Cells on the same anti-diagonal are mutually independent, making
                // this loop structure the natural unit of parallelism in GPU versions.
                //
                // Cyclic buffer arithmetic:
                //   curr_k   → slot for diagonal k   (the one being written)
                //   pre_k    → slot for diagonal k-1 (read by states I and D)
                //   prepre_k → slot for diagonal k-2 (read by state M)
                // ---------------------------------------------------------------
                for (int k = 0; k <= refLen + qryLen; ++k) {

                    int curr_k   = (k     % 3) * (T_TILE + 1);  // write slot
                    int pre_k    = ((k+2) % 3) * (T_TILE + 1);  // diagonal k-1
                    int prepre_k = ((k+1) % 3) * (T_TILE + 1);  // diagonal k-2

                    // Bounds of the i-index for this anti-diagonal
                    // (j = k-i must remain in [0, qryLen])
                    int i_start = max(0,           k - (int)qryLen);
                    int i_end   = min((int)refLen, k);

                    for (int i = i_start; i <= i_end; ++i) {
                        int j = k - i;

                        // Local scores for the three HMM states at cell (i,j)
                        double  vm = NEG_INF_D, vi = NEG_INF_D, vd = NEG_INF_D;
                        // Predecessor state that produced the best score for each state
                        uint8_t pre_m = (uint8_t)S_M;
                        uint8_t pre_i = (uint8_t)S_M;
                        uint8_t pre_d = (uint8_t)S_M;

                        if (i == 0 && j == 0) {
                            // Origin: seed from carry-in
                            // V_s[0][0] = carryLogProb + log π(s)
                            vm = carryLogProb + LOG_INIT_M;
                            vi = carryLogProb + LOG_INIT_I;
                            vd = carryLogProb + LOG_INIT_D;
                        }
                        else if (i == 0) {
                            // Top boundary (i=0, j>0): only query gaps are possible.
                            // Only state I is active — ref has not started yet.
                            // V_I[0][j] = max(V_M[0][j-1] + logT[M→I], V_I[0][j-1] + logT[I→I])
                            double fMI = wf_M[pre_k + 0] + LOG_MI;
                            double fII = wf_I[pre_k + 0] + LOG_II;
                            if (fMI >= fII) { vi = fMI; pre_i = (uint8_t)S_M; }
                            else            { vi = fII; pre_i = (uint8_t)S_I; }
                            d_tbDir  [S_I * CELL + 0 * T_TILE + (j-1)] = DIR_LEFT;
                            d_tbState[S_I * CELL + 0 * T_TILE + (j-1)] = pre_i;
                        }
                        else if (j == 0) {
                            // Left boundary (i>0, j=0): only ref gaps are possible.
                            // Only state D is active — query has not started yet.
                            // V_D[i][0] = max(V_M[i-1][0] + logT[M→D], V_D[i-1][0] + logT[D→D])
                            double fMD = wf_M[pre_k + (i-1)] + LOG_MD;
                            double fDD = wf_D[pre_k + (i-1)] + LOG_DD;
                            if (fMD >= fDD) { vd = fMD; pre_d = (uint8_t)S_M; }
                            else            { vd = fDD; pre_d = (uint8_t)S_D; }
                            d_tbDir  [S_D * CELL + (i-1) * T_TILE + 0] = DIR_UP;
                            d_tbState[S_D * CELL + (i-1) * T_TILE + 0] = pre_d;
                        }
                        else {
                            // Inner cell: compute all 3 HMM states

                            char   r    = d_seqs[refStart + reference_idx + (i-1)];
                            char   q    = d_seqs[qryStart + query_idx     + (j-1)];
                            double emit = (r == q) ? LOG_EMIT_MATCH : LOG_EMIT_MISMATCH;

                            // State M: diagonal predecessor at (i-1, j-1) on diagonal k-2.
                            // All three predecessor states are allowed (M, I, D).
                            // V_M[i][j] = emit + max_s(V_s[i-1][j-1] + logT[s→M])
                            double mMM = wf_M[prepre_k + (i-1)] + LOG_MM;
                            double mIM = wf_I[prepre_k + (i-1)] + LOG_IM;
                            double mDM = wf_D[prepre_k + (i-1)] + LOG_DM;
                            if (mMM >= mIM && mMM >= mDM) { vm = mMM + emit; pre_m = S_M; }
                            else if (mIM >= mDM)           { vm = mIM + emit; pre_m = S_I; }
                            else                           { vm = mDM + emit; pre_m = S_D; }

                            // State I: leftward predecessor at (i, j-1) on diagonal k-1.
                            // D→I is forbidden to prevent alignment ambiguity.
                            // V_I[i][j] = max(V_M[i][j-1] + logT[M→I], V_I[i][j-1] + logT[I→I])
                            double iMI = wf_M[pre_k + i] + LOG_MI;
                            double iII = wf_I[pre_k + i] + LOG_II;
                            if (iMI >= iII) { vi = iMI; pre_i = S_M; }
                            else            { vi = iII; pre_i = S_I; }

                            // State D: upward predecessor at (i-1, j) on diagonal k-1.
                            // I→D is forbidden to prevent alignment ambiguity.
                            // V_D[i][j] = max(V_M[i-1][j] + logT[M→D], V_D[i-1][j] + logT[D→D])
                            double dMD = wf_M[pre_k + (i-1)] + LOG_MD;
                            double dDD = wf_D[pre_k + (i-1)] + LOG_DD;
                            if (dMD >= dDD) { vd = dMD; pre_d = S_M; }
                            else            { vd = dDD; pre_d = S_D; }
                        }

                        // Commit scores to the current wavefront slot.
                        // Index i within the slot corresponds to the ref position.
                        wf_M[curr_k + i] = vm;
                        wf_I[curr_k + i] = vi;
                        wf_D[curr_k + i] = vd;

                        // Write all 3 per-state TB entries for inner cells.
                        // One direction and one predecessor per state are stored separately
                        // so traceback can read the correct entry for any cur_state.
                        // Matches CPU: TB_dir[idx(s,i,j)] / TB_state[idx(s,i,j)].
                        if (i > 0 && j > 0) {
                            int cell = (i-1) * T_TILE + (j-1);
                            d_tbDir  [S_M * CELL + cell] = DIR_DIAG; d_tbState[S_M * CELL + cell] = pre_m;
                            d_tbDir  [S_I * CELL + cell] = DIR_LEFT; d_tbState[S_I * CELL + cell] = pre_i;
                            d_tbDir  [S_D * CELL + cell] = DIR_UP;   d_tbState[S_D * CELL + cell] = pre_d;
                        }

                        // GACT overlap tracking.
                        // Matches CPU exactly: iterate all 3 states with >=,
                        // so last-wins on ties (S_D beats S_I beats S_M on tie).
                        // Only active on non-final tiles; the last tile uses the
                        // terminal cell (refLen, qryLen) as the traceback start.
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

                // ---------------------------------------------------------------
                // TRACEBACK
                // ---------------------------------------------------------------

                // For a non-final tile, start from the best overlap cell.
                // For the final tile, start from the bottom-right corner (refLen, qryLen).
                int ti = (!lastTile) ? (int)best_ti : (int)refLen;
                int tj = (!lastTile) ? (int)best_tj : (int)qryLen;

                // Record how far to advance the sequence cursors after this tile
                int next_ref_advance = ti;
                int next_qry_advance = tj;

                // Update carry-in score for next tile.
                // Non-final tile: carry the best overlap score into the next tile's origin.
                // Final tile: compute the max over all three terminal states (not needed
                // for further tiles, but kept consistent with the CPU implementation).
                carryLogProb = (!lastTile) ? bestOverlapScore : NEG_INF_D;
                if (lastTile) {
                    // Read scores from the wavefront buffer at the terminal anti-diagonal
                    int curr_k = (((int)refLen + (int)qryLen) % 3) * (T_TILE + 1);
                    double fM = wf_M[curr_k + refLen];
                    double fI = wf_I[curr_k + refLen];
                    double fD = wf_D[curr_k + refLen];
                    carryLogProb = fM;
                    if (fI > carryLogProb) carryLogProb = fI;
                    if (fD > carryLogProb) carryLogProb = fD;
                }

                int localLen = 0;

                // Determine starting state for traceback:
                //   non-last tile → state that was best at the overlap cell
                //   last tile     → argmax over the 3 states at (refLen, qryLen)
                int cur_state;
                if (!lastTile) {
                    cur_state = (int)best_state;
                } else {
                    int curr_k = (((int)refLen + (int)qryLen) % 3) * (T_TILE + 1);
                    double fM = wf_M[curr_k + refLen];
                    double fI = wf_I[curr_k + refLen];
                    double fD = wf_D[curr_k + refLen];
                    cur_state = S_M;
                    if (fI > fM && fI > fD) cur_state = S_I;
                    else if (fD > fM)        cur_state = S_D;
                }

                // Walk backwards from (ti, tj) to (0, 0).
                // At each step, cur_state indexes the correct row of d_tbDir/d_tbState.
                // The direction determines how to decrement ti/tj.
                // The predecessor state updates cur_state for the next step.
                // Directions are collected in reverse order into localPath[].
                // Traceback walk — index TB tables by cur_state.
                // Matches CPU: TB_dir[idx(cur_state, ti, tj)].
                while (ti > 0 || tj > 0) {
                    uint8_t dir;
                    int     prev_state;

                    if (ti == 0) {
                        // Top boundary — forced DIR_LEFT, must be in state I
                        dir        = DIR_LEFT;
                        prev_state = (int)d_tbState[S_I * CELL + 0 * T_TILE + (tj-1)];
                        tj--;
                    } else if (tj == 0) {
                        // Left boundary — forced DIR_UP, must be in state D
                        dir        = DIR_UP;
                        prev_state = (int)d_tbState[S_D * CELL + (ti-1) * T_TILE + 0];
                        ti--;
                    } else {
                        int cell   = (ti-1) * T_TILE + (tj-1);
                        dir        = d_tbDir  [cur_state * CELL + cell];
                        prev_state = d_tbState[cur_state * CELL + cell];
                        if      (dir == DIR_DIAG) { ti--; tj--; }
                        else if (dir == DIR_UP)   { ti--;        }
                        else                      {       tj--;  }
                    }

                    localPath[localLen++] = dir;  // collected in reverse
                    cur_state = prev_state;        // follow the predecessor state chain
                }

                // Reverse localPath into d_tb — the global buffer stores paths forward
                for (int k = localLen - 1; k >= 0; --k) {
                    d_tb[tbGlobalOffset + currentPairPathLen] = localPath[k];
                    currentPairPathLen++;
                }

                // Advance sequence cursors to the start of the next tile
                reference_idx += next_ref_advance;
                query_idx     += next_qry_advance;

            } // end tile loop
        } // end pair loop
    } // end bx==0 && tx==0 guard
}

// =============================================================================
// getAlignedSequences
//
// Reconstructs the gapped alignment strings from the compact traceback path.
// Reads direction codes from tb_paths and builds aln0 (reference aligned) and
// aln1 (query aligned) by consuming sequence bases or inserting gap characters:
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
            else break;
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
// Top-level orchestration for the single-thread GPU baseline:
//   1. Allocate GPU memory
//   2. Transfer sequences and metadata to the device
//   3. Launch the kernel (1 block × 1 thread)
//   4. Transfer traceback paths back to host
//   5. Reconstruct aligned strings
//
// Note: cudaDeviceSynchronize() is called after transferTB2Host() to ensure
// the device has fully completed before the host reads the results.
// =============================================================================
void GpuAligner::alignment() {
    int numBlocks = 1;  // single block — only one thread will run
    int blockSize = 1;  // single thread — all pairs processed serially

    allocateMem();
    transferSequence2Device();

    alignmentOnGPU<<<numBlocks, blockSize>>>(d_info, d_seqLen, d_seqs, d_tb,
                                             d_tbDir, d_tbState);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    TB_PATH tb_paths = transferTB2Host();
    cudaDeviceSynchronize();

    getAlignedSequences(tb_paths);
}

// =============================================================================
// writeAlignment
//
// Writes all aligned sequences to a FASTA file.
// If append=true, opens in append mode (for multi-batch output to one file).
// Otherwise creates/overwrites the file.
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
