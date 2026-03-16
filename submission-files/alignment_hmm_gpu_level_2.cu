#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <math.h>

// =============================================================================
// CUDA error checking macro
//
// Wraps any CUDA API call. On failure prints file, line, and error string
// to stderr then exits. Usage: CUDA_CHECK(cudaMalloc(...));
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

// Number of threads per block — must match the blockSize passed to the kernel launch.
// 256 threads = 8 warps per block. On the A30 (56 SMs, 48 KB shared mem/block),
// this yields 2 blocks/SM, giving 512 threads/SM active at once.
#define BLOCK_SIZE 256

// =============================================================================
// allocateMem  — identical to HMM 1-thread
//
// Same allocations as Level 1: d_tbDir and d_tbState are numPairs * 3 * T * T
// so each block has its own private traceback scratch region.
// =============================================================================
void GpuAligner::allocateMem() {
    longestLen = std::max_element(seqs.begin(), seqs.end(),
        [](const Sequence& a, const Sequence& b) {
            return a.seq.size() < b.seq.size();
        })->seq.size();

    cudaError_t err;

    // Flat packed sequence array: all sequences zero-padded to longestLen bytes
    err = cudaMalloc(&d_seqs, numPairs * 2 * longestLen * sizeof(char));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Actual (unpadded) lengths — 2 per pair (ref, qry)
    err = cudaMalloc(&d_seqLen, numPairs * 2 * sizeof(int32_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Output traceback buffer: worst-case 2*longestLen direction codes per pair
    int tb_length = longestLen << 1;
    err = cudaMalloc(&d_tb, numPairs * tb_length * sizeof(uint8_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Two-element info array passed into the kernel: [numPairs, longestLen]
    err = cudaMalloc(&d_info, 2 * sizeof(int32_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Each pair (block) needs its own 3*T*T scratch — per NW parallel pattern
    err = cudaMalloc(&d_tbDir,   (size_t)numPairs * 3 * T_TILE * T_TILE * sizeof(uint8_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    err = cudaMalloc(&d_tbState, (size_t)numPairs * 3 * T_TILE * T_TILE * sizeof(uint8_t));
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }
}

// =============================================================================
// transferSequence2Device  — identical to HMM 1-thread
//
// Packs sequences into flat device arrays and zero-initialises d_tb.
// =============================================================================
void GpuAligner::transferSequence2Device() {
    cudaError_t err;

    // Build flat host array: each sequence zero-padded to longestLen
    std::vector<char> h_seqs(longestLen * numPairs * 2, 0);
    for (size_t i = 0; i < (size_t)(numPairs * 2); ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(h_seqs.data() + (i * longestLen), s.data(), s.size());
    }
    err = cudaMemcpy(d_seqs, h_seqs.data(),
                     longestLen * numPairs * 2 * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Copy actual sequence lengths
    std::vector<int32_t> h_seqLen(numPairs * 2, 0);
    for (int i = 0; i < numPairs * 2; ++i) h_seqLen[i] = seqs[i].seq.size();
    err = cudaMemcpy(d_seqLen, h_seqLen.data(),
                     numPairs * 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Zero-initialise traceback buffer — 0 is the end-of-path sentinel
    int tb_length = longestLen << 1;
    std::vector<uint8_t> h_tb(tb_length * numPairs, 0);
    err = cudaMemcpy(d_tb, h_tb.data(),
                     tb_length * numPairs * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }

    // Copy info array: [numPairs, longestLen]
    std::vector<int32_t> h_info(2);
    h_info[0] = numPairs;
    h_info[1] = longestLen;
    err = cudaMemcpy(d_info, h_info.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }
}

// =============================================================================
// transferTB2Host  — identical to HMM 1-thread
//
// Copies the completed traceback paths from GPU to host for getAlignedSequences.
// =============================================================================
TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen << 1;
    TB_PATH h_tb(tb_length * numPairs);
    cudaError_t err = cudaMemcpy(h_tb.data(), d_tb,
                                 tb_length * numPairs * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "GPU_ERROR: %s\n", cudaGetErrorString(err)); exit(1); }
    return h_tb;
}

// =============================================================================
// HMM log-probability constants — identical to HMM 1-thread
// =============================================================================
#define LOG_MM            -0.10536051565782630   // log(0.90)  M→M
#define LOG_MI            -2.99573227355399099   // log(0.05)  M→I
#define LOG_MD            -2.99573227355399099   // log(0.05)  M→D
#define LOG_IM            -0.35667494393873245   // log(0.70)  I→M
#define LOG_II            -1.20397280432593597   // log(0.30)  I→I
#define LOG_DM            -0.35667494393873245   // log(0.70)  D→M
#define LOG_DD            -1.20397280432593597   // log(0.30)  D→D
#define LOG_INIT_M        -0.10536051565782630   // log(0.90)  initial prob of state M
#define LOG_INIT_I        -2.99573227355399099   // log(0.05)  initial prob of state I
#define LOG_INIT_D        -2.99573227355399099   // log(0.05)  initial prob of state D
#define LOG_EMIT_MATCH    -0.10536051565782630   // log(0.90)  emission: ref[i] == qry[j]
#define LOG_EMIT_MISMATCH -2.30258509299404568   // log(0.10)  emission: ref[i] != qry[j]
#define NEG_INF_D         -1e18                  // log(0) — unreachable state sentinel

// T_TILE defined in alignment.cuh (= 200)
// O_TILE: size of the overlap corner scanned at end of each non-final tile
#define O_TILE 64

// =============================================================================
// alignmentOnGPU — Level 1 + Level 2 parallelism
//
// LEVEL 1 — Inter-pair parallelism (same as Level 1 file):
//   pair = blockIdx.x. All pairs run simultaneously, one block each.
//
// LEVEL 2 — Intra-pair wavefront parallelism (NEW in this version):
//   Within each block, the 256 threads split the cells of each anti-diagonal
//   among themselves. Thread tx handles cells i = i_start + tx, i_start + tx
//   + blockDim.x, ... on diagonal k. A __syncthreads() after each diagonal
//   ensures all writes to wf_M/I/D are visible before the next diagonal reads.
//
// SHARED MEMORY ADDITIONS OVER LEVEL 1
// -------------------------------------
//   shared_ref / shared_qry : tile's sequence segments loaded cooperatively
//     once per tile, then read from fast shared memory by all threads.
//   s_bestScore / s_best_ti / s_best_tj / s_best_st : per-thread overlap
//     accumulators written after the wavefront loop, then tree-reduced to find
//     the global best overlap cell for the tile.
//   sh_* tile-boundary scalars : promoted from thread-local registers (Level 1)
//     to __shared__ so all 256 threads can read them each iteration.
//
// TILE-BOUNDARY SCALARS MUST NOW BE __shared__
// ---------------------------------------------
//   In Level 1 there was only one thread per block, so scalars like lastTile
//   and carryLogProb lived in registers. With 256 threads, every thread needs
//   to read them — they are written by tx==0 and broadcast via shared memory +
//   __syncthreads(), the same pattern used in the NW GPU reference.
//
// OVERLAP REDUCTION
// -----------------
//   Each thread accumulates its own local best (localBestScore/ti/tj/st) in
//   registers during the wavefront loop — no shared memory contention. After
//   the loop, all threads write their local best to s_bestScore[tx] etc., then
//   a standard log-2 tree reduction finds the global best in O(log BLOCK_SIZE)
//   steps. The tie-breaking rule (>= with state priority S_D > S_I > S_M) must
//   match the single-thread sequential behaviour exactly to preserve correctness.
//
// TRACEBACK AND PATH WRITE
// ------------------------
//   Traceback is serial (tx==0 only) — each step depends on the previous.
//   Once tx==0 fills localPath[] in reverse, all 256 threads cooperate on a
//   coalesced write to d_tb by striding: for(k=tx; k<localLen; k+=blockDim.x).
// =============================================================================

__global__ void alignmentOnGPU(
    int32_t* d_info,       // [0]=numPairs, [1]=longestLen (sequence stride)
    int32_t* d_seqLen,     // actual lengths, 2 entries per pair
    char*    d_seqs,       // flat packed sequences, stride=longestLen
    uint8_t* d_tb,         // output: traceback direction codes per pair
    uint8_t* d_tbDir,      // global scratch: numPairs * 3 * T_TILE * T_TILE
    uint8_t* d_tbState)    // global scratch: numPairs * 3 * T_TILE * T_TILE
{
    // Level 1: one block per pair
    const int pair = blockIdx.x;
    const int tx   = threadIdx.x;

    if (pair >= d_info[0]) return;

    const uint8_t DIR_DIAG = 1;   // match/mismatch: both sequences advance
    const uint8_t DIR_UP   = 2;   // deletion: only ref advances (gap in query)
    const uint8_t DIR_LEFT = 3;   // insertion: only query advances (gap in ref)
    const int S_M = 0, S_I = 1, S_D = 2;
    const int CELL = T_TILE * T_TILE;  // stride between state planes in tbDir/tbState

    const int32_t maxSeqLen = d_info[1];  // = longestLen, stride in d_seqs

    // Per-pair scratch in global memory — offset by pair (same as Level 1)
    uint8_t* tbDir   = d_tbDir   + (size_t)pair * 3 * CELL;
    uint8_t* tbState = d_tbState + (size_t)pair * 3 * CELL;

    // ------------------------------------------------------------------
    // Shared memory
    // ------------------------------------------------------------------
    // Wavefront score buffers (cyclic triple-buffer)
    // Each holds T_TILE+1 doubles for 3 diagonal slots — ~14.4 KB total
    __shared__ double  wf_M[3 * (T_TILE + 1)];
    __shared__ double  wf_I[3 * (T_TILE + 1)];
    __shared__ double  wf_D[3 * (T_TILE + 1)];

    // Sequence cache — loaded once per tile (same as NW shared_ref/shared_qry)
    // Avoids repeated global memory reads: all threads read from fast shared mem
    __shared__ char    shared_ref[T_TILE];
    __shared__ char    shared_qry[T_TILE];

    // Traceback path buffer for this tile (written by tx==0, read by all for coalesced write)
    __shared__ uint8_t localPath[2 * T_TILE];

    // Per-thread overlap accumulator arrays —
    // same pattern as NW s_maxScore[BLOCK_SIZE]/s_best_i/s_best_j
    // Written after wavefront loop; consumed by tree reduction
    __shared__ double  s_bestScore[BLOCK_SIZE];
    __shared__ int32_t s_best_ti  [BLOCK_SIZE];
    __shared__ int32_t s_best_tj  [BLOCK_SIZE];
    __shared__ int32_t s_best_st  [BLOCK_SIZE];

    // Tile-boundary scalars — same pattern as NW sh_* variables
    // Written by tx==0, read by all threads. Must be __shared__ (unlike Level 1
    // where they were thread-local registers) because all 256 threads need them.
    __shared__ bool    sh_lastTile;           // true when this tile reaches end of both sequences
    __shared__ double  sh_carryLogProb;       // best overlap score seeding next tile's (0,0)
    __shared__ int32_t sh_reference_idx;      // current tile start in the reference
    __shared__ int32_t sh_query_idx;          // current tile start in the query
    __shared__ int32_t sh_currentPairPathLen; // total direction codes written so far for this pair
    __shared__ int32_t sh_localLen;           // number of steps in this tile's traceback path
    __shared__ int32_t sh_next_ref_advance;   // how far to advance reference_idx after this tile
    __shared__ int32_t sh_next_qry_advance;   // how far to advance query_idx after this tile
    __shared__ int32_t sh_best_state;         // final best state after reduction

    // Init shared scalars — tx==0 only, then sync (same as NW)
    if (tx == 0) {
        sh_lastTile           = false;
        sh_carryLogProb       = 0.0;   // log(1) — neutral seed for the first tile
        sh_reference_idx      = 0;
        sh_query_idx          = 0;
        sh_currentPairPathLen = 0;
    }
    __syncthreads();

    // Byte offsets into d_seqs and d_tb for this pair — computed once, used every tile
    const int32_t refStart       = (pair * 2)     * maxSeqLen;  // reference sequence start
    const int32_t qryStart       = (pair * 2 + 1) * maxSeqLen;  // query sequence start
    const int32_t tbGlobalOffset = pair * (maxSeqLen * 2);       // traceback path start in d_tb
    const int32_t refTotalLen    = d_seqLen[2 * pair];           // full reference length
    const int32_t qryTotalLen    = d_seqLen[2 * pair + 1];       // full query length

    // -----------------------------------------------------------------------
    // TILE LOOP
    // -----------------------------------------------------------------------
    while (!sh_lastTile) {

        // All threads read tile-boundary scalars into registers for this iteration
        const int32_t reference_idx = sh_reference_idx;
        const int32_t query_idx     = sh_query_idx;
        const double  carryLogProb  = sh_carryLogProb;

        // Clamp tile dimensions to remaining sequence lengths
        const int32_t refLen = min(T_TILE, (int)(refTotalLen - reference_idx));
        const int32_t qryLen = min(T_TILE, (int)(qryTotalLen - query_idx));

        // tx==0 sets lastTile — same as NW
        if (tx == 0) {
            if ((reference_idx + refLen == refTotalLen) &&
                (query_idx     + qryLen == qryTotalLen))
                sh_lastTile = true;
        }
        __syncthreads();   // all threads must see sh_lastTile before overlap tracking uses it

        // Init wf buffers — strided across all threads, same as NW wf_scores init
        // Each thread initialises every (BLOCK_SIZE)-th element for efficiency
        for (int s = tx; s < 3 * (T_TILE + 1); s += blockDim.x) {
            wf_M[s] = NEG_INF_D;
            wf_I[s] = NEG_INF_D;
            wf_D[s] = NEG_INF_D;
        }
        __syncthreads();

        // Load sequence segments into shared memory — same as NW shared_ref/shared_qry load
        // All threads cooperate so each thread loads one element
        for (int s = tx; s < refLen; s += blockDim.x)
            shared_ref[s] = d_seqs[refStart + reference_idx + s];
        for (int s = tx; s < qryLen; s += blockDim.x)
            shared_qry[s] = d_seqs[qryStart + query_idx + s];
        __syncthreads();

        // Per-thread local overlap best — registers, same as NW localMaxScore/localBest_i/j
        // Each thread tracks the best cell it personally computed — no shared memory writes
        // during the hot wavefront loop. Results are merged by reduction afterward.
        double  localBestScore = NEG_INF_D;
        int32_t localBest_ti   = refLen;
        int32_t localBest_tj   = qryLen;
        int32_t localBest_st   = S_M;

        // -------------------------------------------------------------------
        // WAVEFRONT (ANTI-DIAGONAL) LOOP
        // Level 2: each thread handles one or more cells per diagonal
        // using "i = i_start + tx; i <= i_end; i += blockDim.x"
        // exactly as in NW.
        // __syncthreads() at end of each diagonal — same as NW.
        //
        // WHY SYNC AFTER EVERY DIAGONAL:
        //   Cell (i,j) on diagonal k reads wf_M/I/D values from diagonals k-1
        //   and k-2. Those values were written by potentially different threads
        //   on the previous iteration. Without a barrier, a thread could start
        //   reading diagonal k-1 before another thread has finished writing it.
        // -------------------------------------------------------------------
        for (int k = 0; k <= refLen + qryLen; ++k) {

            const int curr_k   = (k     % 3) * (T_TILE + 1);  // write slot for diagonal k
            const int pre_k    = ((k+2) % 3) * (T_TILE + 1);  // diagonal k-1
            const int prepre_k = ((k+1) % 3) * (T_TILE + 1);  // diagonal k-2

            // Valid i range for this anti-diagonal (j = k-i must be in [0, qryLen])
            const int i_start = max(0,           k - (int)qryLen);
            const int i_end   = min((int)refLen, k);

            // Level 2: stride across diagonal cells — same as NW
            // Thread tx handles i = i_start+tx, i_start+tx+256, ...
            for (int i = i_start + tx; i <= i_end; i += blockDim.x) {
                int j = k - i;

                // ---- vvv HMM SCORING — VERBATIM FROM HMM 1-THREAD vvv ----
                double  vm = NEG_INF_D, vi = NEG_INF_D, vd = NEG_INF_D;
                uint8_t pre_m = (uint8_t)S_M;
                uint8_t pre_i = (uint8_t)S_M;
                uint8_t pre_d = (uint8_t)S_M;

                if (i == 0 && j == 0) {
                    // Tile origin: seed all three states from the carry-in score
                    vm = carryLogProb + LOG_INIT_M;
                    vi = carryLogProb + LOG_INIT_I;
                    vd = carryLogProb + LOG_INIT_D;
                }
                else if (i == 0) {
                    // Top boundary: only I active (query gap)
                    double fMI = wf_M[pre_k + 0] + LOG_MI;
                    double fII = wf_I[pre_k + 0] + LOG_II;
                    if (fMI >= fII) { vi = fMI; pre_i = (uint8_t)S_M; }
                    else            { vi = fII; pre_i = (uint8_t)S_I; }
                    tbDir  [S_I * CELL + 0 * T_TILE + (j-1)] = DIR_LEFT;
                    tbState[S_I * CELL + 0 * T_TILE + (j-1)] = pre_i;
                }
                else if (j == 0) {
                    // Left boundary: only D active (ref gap)
                    double fMD = wf_M[pre_k + (i-1)] + LOG_MD;
                    double fDD = wf_D[pre_k + (i-1)] + LOG_DD;
                    if (fMD >= fDD) { vd = fMD; pre_d = (uint8_t)S_M; }
                    else            { vd = fDD; pre_d = (uint8_t)S_D; }
                    tbDir  [S_D * CELL + (i-1) * T_TILE + 0] = DIR_UP;
                    tbState[S_D * CELL + (i-1) * T_TILE + 0] = pre_d;
                }
                else {
                    // Use shared memory sequence cache (same as NW shared_ref/shared_qry)
                    char   r    = shared_ref[i-1];
                    char   q    = shared_qry[j-1];
                    double emit = (r == q) ? LOG_EMIT_MATCH : LOG_EMIT_MISMATCH;

                    // State M: diagonal predecessor at (i-1,j-1) on diagonal k-2
                    double mMM = wf_M[prepre_k + (i-1)] + LOG_MM;
                    double mIM = wf_I[prepre_k + (i-1)] + LOG_IM;
                    double mDM = wf_D[prepre_k + (i-1)] + LOG_DM;
                    if (mMM >= mIM && mMM >= mDM) { vm = mMM + emit; pre_m = S_M; }
                    else if (mIM >= mDM)           { vm = mIM + emit; pre_m = S_I; }
                    else                           { vm = mDM + emit; pre_m = S_D; }

                    // State I: leftward predecessor at (i,j-1) on diagonal k-1. D→I forbidden.
                    double iMI = wf_M[pre_k + i] + LOG_MI;
                    double iII = wf_I[pre_k + i] + LOG_II;
                    if (iMI >= iII) { vi = iMI; pre_i = S_M; }
                    else            { vi = iII; pre_i = S_I; }

                    // State D: upward predecessor at (i-1,j) on diagonal k-1. I→D forbidden.
                    double dMD = wf_M[pre_k + (i-1)] + LOG_MD;
                    double dDD = wf_D[pre_k + (i-1)] + LOG_DD;
                    if (dMD >= dDD) { vd = dMD; pre_d = S_M; }
                    else            { vd = dDD; pre_d = S_D; }

                    // Write traceback entries for all three states at this inner cell
                    int cell = (i-1) * T_TILE + (j-1);
                    tbDir  [S_M * CELL + cell] = DIR_DIAG; tbState[S_M * CELL + cell] = pre_m;
                    tbDir  [S_I * CELL + cell] = DIR_LEFT; tbState[S_I * CELL + cell] = pre_i;
                    tbDir  [S_D * CELL + cell] = DIR_UP;   tbState[S_D * CELL + cell] = pre_d;
                }

                // Commit scores to the current wavefront slot
                wf_M[curr_k + i] = vm;
                wf_I[curr_k + i] = vi;
                wf_D[curr_k + i] = vd;
                // ---- ^^^ END VERBATIM HMM SCORING ^^^ ----

                // Update thread-local overlap best — registers only, same as NW localMaxScore
                // Uses >= with s going 0->2, so on equal score S_D(2) beats S_I(1) beats S_M(0)
                // exactly matching the 1-thread single-pass behaviour.
                if (!sh_lastTile &&
                    i > (refLen - O_TILE) && j > (qryLen - O_TILE)) {
                    double scores[3] = {vm, vi, vd};
                    for (int s = 0; s < 3; ++s) {
                        if (scores[s] >= localBestScore) {
                            localBestScore = scores[s];
                            localBest_st   = s;
                            localBest_ti   = i;
                            localBest_tj   = j;
                        }
                    }
                }

            } // end inner i loop

            // Sync after each diagonal — same as NW
            // Mandatory: next diagonal reads wf values written by this diagonal
            __syncthreads();

        } // end wavefront loop

        // -------------------------------------------------------------------
        // OVERLAP REDUCTION — same pattern as NW s_maxScore reduction
        //
        // Step 1: each thread writes its local best to shared arrays
        // Step 2: __syncthreads()
        // Step 3: tree reduction — each round halves the active thread count
        //         using a stride that starts at BLOCK_SIZE/2 and shifts right
        // Step 4: tx==0 writes result to sh_next_ref/qry_advance and sh_best_state
        // Step 5: __syncthreads()
        //
        // TIE-BREAKING:
        //   The 1-thread loop iterates states s=0,1,2 with >=, meaning the last
        //   state seen wins on a tie: S_D(2) > S_I(1) > S_M(0). The reduction
        //   must replicate this by preferring higher s_best_st on equal scores.
        // -------------------------------------------------------------------
        s_bestScore[tx] = localBestScore;
        s_best_ti  [tx] = localBest_ti;
        s_best_tj  [tx] = localBest_tj;
        s_best_st  [tx] = localBest_st;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tx < stride) {
                // Match 1-thread tie-break exactly:
                // The 1-thread loop iterates s=0,1,2 with >=, so on equal scores
                // the last state wins: S_D(2) > S_I(1) > S_M(0).
                // In the reduction, on equal scores we prefer higher s_best_st.
                bool take = (s_bestScore[tx + stride] > s_bestScore[tx]) ||
                            (s_bestScore[tx + stride] == s_bestScore[tx] &&
                             s_best_st[tx + stride] >= s_best_st[tx]);
                if (take) {
                    s_bestScore[tx] = s_bestScore[tx + stride];
                    s_best_ti  [tx] = s_best_ti  [tx + stride];
                    s_best_tj  [tx] = s_best_tj  [tx + stride];
                    s_best_st  [tx] = s_best_st  [tx + stride];
                }
            }
            __syncthreads();
        }

        // tx==0 writes the reduced result — same as NW tx==0 writing sh_next_ref/qry_advance
        // After this, s_bestScore[0]/s_best_ti[0]/s_best_tj[0]/s_best_st[0] hold the global best
        if (tx == 0 && !sh_lastTile) {
            sh_next_ref_advance = s_best_ti[0];
            sh_next_qry_advance = s_best_tj[0];
            sh_best_state       = s_best_st[0];
        }
        __syncthreads();

        // -------------------------------------------------------------------
        // TRACEBACK — tx==0 only, verbatim from HMM 1-thread
        //
        // Traceback is inherently serial (each step depends on the previous),
        // so only thread 0 executes it. Other threads wait at the syncthreads
        // below. The resulting path is stored in localPath[] (reversed), then
        // all threads cooperate on the coalesced write to d_tb.
        // -------------------------------------------------------------------
        if (tx == 0) {
            sh_localLen = 0;

            // Start from best overlap cell (non-final tile) or terminal cell (final tile)
            int ti = (!sh_lastTile) ? sh_next_ref_advance : (int)refLen;
            int tj = (!sh_lastTile) ? sh_next_qry_advance : (int)qryLen;

            // Lock in the advance distances before traceback modifies ti/tj
            sh_next_ref_advance = ti;
            sh_next_qry_advance = tj;

            // Update carry-in for next tile — verbatim from HMM 1-thread
            double carryOut = (!sh_lastTile) ? s_bestScore[0] : NEG_INF_D;
            if (sh_lastTile) {
                int curr_k = (((int)refLen + (int)qryLen) % 3) * (T_TILE + 1);
                double fM = wf_M[curr_k + refLen];
                double fI = wf_I[curr_k + refLen];
                double fD = wf_D[curr_k + refLen];
                carryOut = fM;
                if (fI > carryOut) carryOut = fI;
                if (fD > carryOut) carryOut = fD;
            }
            sh_carryLogProb = carryOut;

            // Determine starting state — verbatim from HMM 1-thread
            int cur_state;
            if (!sh_lastTile) {
                cur_state = sh_best_state;
            } else {
                int curr_k = (((int)refLen + (int)qryLen) % 3) * (T_TILE + 1);
                double fM = wf_M[curr_k + refLen];
                double fI = wf_I[curr_k + refLen];
                double fD = wf_D[curr_k + refLen];
                cur_state = S_M;
                if (fI > fM && fI > fD) cur_state = S_I;
                else if (fD > fM)        cur_state = S_D;
            }

            // Traceback walk — verbatim from HMM 1-thread
            while (ti > 0 || tj > 0) {
                uint8_t dir;
                int     prev_state;

                if (ti == 0) {
                    dir        = DIR_LEFT;
                    prev_state = (int)tbState[S_I * CELL + 0 * T_TILE + (tj-1)];
                    tj--;
                } else if (tj == 0) {
                    dir        = DIR_UP;
                    prev_state = (int)tbState[S_D * CELL + (ti-1) * T_TILE + 0];
                    ti--;
                } else {
                    int cell   = (ti-1) * T_TILE + (tj-1);
                    dir        = tbDir  [cur_state * CELL + cell];
                    prev_state = tbState[cur_state * CELL + cell];
                    if      (dir == DIR_DIAG) { ti--; tj--; }
                    else if (dir == DIR_UP)   { ti--;        }
                    else                      {       tj--;  }
                }

                localPath[sh_localLen++] = dir;
                cur_state = prev_state;
            }
        } // end tx==0 traceback
        __syncthreads();

        // Coalesced write of reversed path to global memory — same as NW
        // All 256 threads write different elements of localPath[] simultaneously
        int localLen             = sh_localLen;
        int32_t currentPathStart = sh_currentPairPathLen;
        for (int k = tx; k < localLen; k += blockDim.x) {
            d_tb[tbGlobalOffset + currentPathStart + k] = localPath[localLen - 1 - k];
        }
        __syncthreads();

        // tx==0 advances tile-boundary scalars — same as NW
        if (tx == 0) {
            sh_currentPairPathLen += sh_localLen;
            sh_reference_idx      += sh_next_ref_advance;
            sh_query_idx          += sh_next_qry_advance;
        }
        __syncthreads();

    } // end tile loop
}

// =============================================================================
// getAlignedSequences — identical to HMM 1-thread
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
// clearAndReset — identical to HMM 1-thread
//
// Frees all GPU device memory and resets the aligner for the next batch.
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
// alignment — numPairs blocks, BLOCK_SIZE threads each
//
// Launches numPairs blocks of 256 threads. Each block handles one pair
// (Level 1) and within each block all 256 threads collaborate on the
// wavefront anti-diagonals (Level 2).
// =============================================================================
void GpuAligner::alignment() {
    const int numBlocks = numPairs;
    const int blockSize = BLOCK_SIZE;   // must match BLOCK_SIZE in kernel

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
// writeAlignment — identical to HMM 1-thread
//
// Writes all aligned sequences to a FASTA file.
// append=true for multi-batch output to the same file.
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
