/*
 * ============================================================================
 * HMM-BASED PAIRWISE SEQUENCE ALIGNMENT (GPU/CUDA) — FULLY PARALLELIZED
 * ============================================================================
 *
 * PARALLELIZATION STRATEGY (mirrors optimized NW in alignment_nw_parallel.cu):
 *
 *  1. ONE BLOCK PER PAIR   — blockIdx.x selects the sequence pair.
 *                            All pairs run concurrently across the GPU.
 *
 *  2. WAVEFRONT PARALLELISM — For diagonal k, cells (i, k-i) are independent.
 *                             Each thread handles one or more cells via striding:
 *                               i = i_start + tx, i += blockDim.x
 *                             This exactly mirrors the NW strided wavefront loop.
 *
 *  3. SHARED MEMORY LOADS  — Tile segments shared_ref[T] and shared_qry[T] are
 *                             loaded cooperatively with stride blockDim.x, giving
 *                             coalesced global reads.
 *
 *  4. PARALLEL REDUCTION   — After the wavefront loop, per-thread local overlap
 *                             maxima (localMaxScore_M/IX/IY, localBest_i/j) are
 *                             written into shared arrays s_maxScore/s_best_i/j
 *                             then reduced with a log-stride tree — identical to
 *                             the NW reduction.  The Viterbi version takes the
 *                             max over all three HMM states at each cell.
 *
 *  5. COALESCED TB WRITE   — The traceback reversal loop uses strided writes:
 *                               d_tb[offset + pathLen + k]  (k = tx, +=blockDim.x)
 *                             matching the NW coalesced write pattern.
 *
 *  6. SHARED STATE         — Tile-loop cursor variables (sh_reference_idx,
 *                             sh_query_idx, sh_currentPairPathLen, etc.) live in
 *                             shared memory and are updated only by thread 0,
 *                             guarded by __syncthreads(), just like the NW version.
 *
 *  7. DPX INTRINSICS       — On sm_90+ the three-way max in the M-state is
 *                             accelerated with __vimax3_s16x2, matching the NW
 *                             bonus optimisation.
 *
 * HMM recap (3-state Pair HMM, Viterbi decoding):
 *   M  – match/mismatch  (diagonal step)
 *   IX – gap in ref      (left step, query advances)
 *   IY – gap in query    (up step, ref advances)
 *
 * Three separate wavefront arrays wf_M / wf_IX / wf_IY replace the single
 * wf_scores array used in the NW kernel.  Everything else (tile loop, overlap
 * logic, traceback, shared-memory layout) is structurally identical.
 * ============================================================================
 */

#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda.h>

#if __CUDA_ARCH__ >= 900
#include <cuda/dpx_intrinsics.h>
#endif

// ============================================================================
// Host-side memory management  (unchanged)
// ============================================================================

void GpuAligner::allocateMem() {
    longestLen = std::max_element(seqs.begin(), seqs.end(),
        [](const Sequence& a, const Sequence& b){
            return a.seq.size() < b.seq.size();
        })->seq.size();

    cudaError_t err;

    err = cudaMalloc(&d_seqs, numPairs * 2 * longestLen * sizeof(char));
    if (err != cudaSuccess){fprintf(stderr,"GPU_ERROR: %s (%s)\n",cudaGetErrorString(err),cudaGetErrorName(err));exit(1);}

    err = cudaMalloc(&d_seqLen, numPairs * 2 * sizeof(int32_t));
    if (err != cudaSuccess){fprintf(stderr,"GPU_ERROR: %s (%s)\n",cudaGetErrorString(err),cudaGetErrorName(err));exit(1);}

    int tb_length = longestLen << 1;
    err = cudaMalloc(&d_tb, numPairs * tb_length * sizeof(uint8_t));
    if (err != cudaSuccess){fprintf(stderr,"GPU_ERROR: %s (%s)\n",cudaGetErrorString(err),cudaGetErrorName(err));exit(1);}

    err = cudaMalloc(&d_info, 2 * sizeof(int32_t));
    if (err != cudaSuccess){fprintf(stderr,"GPU_ERROR: %s (%s)\n",cudaGetErrorString(err),cudaGetErrorName(err));exit(1);}
}

void GpuAligner::transferSequence2Device() {
    cudaError_t err;

    std::vector<char> h_seqs(longestLen * numPairs * 2, 0);
    for (size_t i = 0; i < numPairs * 2; ++i){
        const std::string& s = seqs[i].seq;
        std::memcpy(h_seqs.data() + (i * longestLen), s.data(), s.size());
    }
    err = cudaMemcpy(d_seqs, h_seqs.data(), longestLen * numPairs * 2 * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){fprintf(stderr,"GPU_ERROR: %s (%s)\n",cudaGetErrorString(err),cudaGetErrorName(err));exit(1);}

    std::vector<int32_t> h_seqLen(numPairs * 2, 0);
    for (int i = 0; i < numPairs * 2; ++i) h_seqLen[i] = seqs[i].seq.size();
    err = cudaMemcpy(d_seqLen, h_seqLen.data(), numPairs * 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){fprintf(stderr,"GPU_ERROR: %s (%s)\n",cudaGetErrorString(err),cudaGetErrorName(err));exit(1);}

    int tb_length = longestLen << 1;
    std::vector<uint8_t> h_tb(tb_length * numPairs, 0);
    err = cudaMemcpy(d_tb, h_tb.data(), tb_length * numPairs * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){fprintf(stderr,"GPU_ERROR: %s (%s)\n",cudaGetErrorString(err),cudaGetErrorName(err));exit(1);}

    std::vector<int32_t> h_info(2);
    h_info[0] = numPairs;
    h_info[1] = longestLen;
    err = cudaMemcpy(d_info, h_info.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){fprintf(stderr,"GPU_ERROR: %s (%s)\n",cudaGetErrorString(err),cudaGetErrorName(err));exit(1);}
}

TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen << 1;
    TB_PATH h_tb(tb_length * numPairs);
    cudaError_t err = cudaMemcpy(h_tb.data(), d_tb,
                                  tb_length * numPairs * sizeof(uint8_t),
                                  cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){fprintf(stderr,"GPU_ERROR: %s (%s)\n",cudaGetErrorString(err),cudaGetErrorName(err));exit(1);}
    return h_tb;
}


// ============================================================================
// CUDA Kernel — Pair HMM Viterbi, fully parallelised
// ============================================================================
__global__ void alignmentOnGPU(
    int32_t* d_info,
    int32_t* d_seqLen,
    char*    d_seqs,
    uint8_t* d_tb
) {
    // -----------------------------------------------------------------------
    // KERNEL CONFIGURATION
    // -----------------------------------------------------------------------
    int bx = blockIdx.x;   // one block per alignment pair
    int tx = threadIdx.x;

    // GACT tile parameters — tune for speed/accuracy tradeoff
    const int T = 200;   // tile size  (must be <= shared memory budget)
    const int O = 64;    // overlap region width

    // Must match blockSize in alignment() below
    const int BLOCK_SIZE = 256;

    // -----------------------------------------------------------------------
    // Pair HMM Viterbi scores (scaled log-probabilities, int16)
    //
    //  Transitions (scaled ×5, rounded):
    //    t_MM   ≈  0   (absorbed into MATCH / MISMATCH emission below)
    //    t_MIX  = -12  gap open  M  → IX or IY
    //    t_IXM  = -10  close gap IX → M
    //    t_IXIX =  -1  extend gap   (IX → IX)
    //    t_IXIY = -15  switch gaps  (IX → IY)
    //    t_IYM  = -10  close gap IY → M  (same as t_IXM by symmetry)
    //    t_IYIX = -15  switch gaps  (IY → IX)
    //    t_IYIY =  -1  extend gap   (IY → IY)
    //
    //  Emissions (scaled ×5, rounded):
    //    e_match    = +8   (ln 0.85 × 5)
    //    e_mismatch = -15  (ln 0.05 × 5)
    //    e_gap      =  -7  (ln 0.25 × 5)
    //
    //  Effective combined scores used in DP:
    //    MATCH       = e_match    + t_MM   =  +8
    //    MISMATCH    = e_mismatch + t_MM   = -15
    //    GAP_OPEN_M  = e_gap      + t_MIX  = -19  (M → IX or IY, includes gap emit)
    //    GAP_EXT     = e_gap      + t_IXIX = -8   (extend any gap state)
    //    CLOSE_GAP   = t_IXM               = -10  (IX/IY → M transition only)
    //    SWITCH_GAP  = e_gap      + t_IXIY = -22  (IX ↔ IY, rare)
    // -----------------------------------------------------------------------
    const int16_t MATCH      =   8;
    const int16_t MISMATCH   = -15;
    const int16_t GAP_OPEN_M = -19;
    const int16_t GAP_EXT    =  -8;
    const int16_t CLOSE_GAP  = -10;
    const int16_t SWITCH_GAP = -22;

    // Traceback direction / state constants
    const uint8_t DIR_DIAG = 1;   // M  state — diagonal step
    const uint8_t DIR_LEFT = 3;   // IX state — left step  (gap in ref)
    const uint8_t DIR_UP   = 2;   // IY state — up step    (gap in query)

    const int16_t NEG_INF = -9999;

    // -----------------------------------------------------------------------
    // Shared Memory
    // -----------------------------------------------------------------------

    // Three wavefront ring-buffers, one per HMM state.
    // Layout: [diag_slot 0..2][cell 0..T] — slot cycles as (k % 3).
    // Size 3*(T+1) identical to the single wf_scores in the NW kernel,
    // but replicated for M, IX, IY states.
    __shared__ int16_t wf_M  [3 * (T + 1)];
    __shared__ int16_t wf_IX [3 * (T + 1)];
    __shared__ int16_t wf_IY [3 * (T + 1)];

    // Traceback direction table for inner cells (T×T).
    // Encodes the winning predecessor direction at each (i,j).
    __shared__ uint8_t tbDir[T * T];

    // Tile-local reversed traceback path, max length 2*T.
    __shared__ uint8_t localPath[2 * T];

    // Tile control flags — updated only by thread 0, broadcast via __syncthreads
    __shared__ bool    lastTile;
    __shared__ int16_t tileStartScore;   // best overlap score carried into next tile origin

    // Per-thread local maxima — written here before parallel reduction.
    // Sized to BLOCK_SIZE so every thread index fits.
    __shared__ int s_maxScore[BLOCK_SIZE];
    __shared__ int s_best_i  [BLOCK_SIZE];
    __shared__ int s_best_j  [BLOCK_SIZE];

    // Tile sequence segments — loaded cooperatively for coalesced access.
    __shared__ char shared_ref[T];
    __shared__ char shared_qry[T];

    // Shared tile-loop cursor variables — updated by thread 0 only.
    __shared__ int32_t sh_localLen;
    __shared__ int32_t sh_currentPairPathLen;
    __shared__ int32_t sh_reference_idx;
    __shared__ int32_t sh_query_idx;
    __shared__ int32_t sh_next_ref_advance;
    __shared__ int32_t sh_next_qry_advance;

    // -----------------------------------------------------------------------
    // Per-block (per-pair) setup
    // -----------------------------------------------------------------------
    int32_t maxSeqLen = d_info[1];
    int pair = bx;   // one block per pair — mirrors NW parallel version

    if (tx == 0) {
        lastTile             = false;
        tileStartScore       = 0;
        sh_currentPairPathLen = 0;
        sh_reference_idx     = 0;
        sh_query_idx         = 0;
    }
    __syncthreads();

    int32_t refStart       = (pair * 2)     * maxSeqLen;
    int32_t qryStart       = (pair * 2 + 1) * maxSeqLen;
    int32_t tbGlobalOffset = pair * (maxSeqLen * 2);

    int32_t refTotalLen = d_seqLen[2 * pair];
    int32_t qryTotalLen = d_seqLen[2 * pair + 1];

    // =========================================================================
    // TILE LOOP
    // =========================================================================
    while (!lastTile) {

        // Register-local snapshot of shared cursors for this tile
        int32_t reference_idx = sh_reference_idx;
        int32_t query_idx     = sh_query_idx;

        int32_t refLen = min(T, refTotalLen - reference_idx);
        int32_t qryLen = min(T, qryTotalLen - query_idx);

        // --- Last-tile detection (thread 0 only, others see result after sync)
        if (tx == 0) {
            if ((reference_idx + refLen == refTotalLen) &&
                (query_idx     + qryLen == qryTotalLen))
                lastTile = true;
        }

        // --- Initialise all three wavefront buffers to NEG_INF ---
        // Parallelised with striding: each thread covers multiple elements.
        // Three separate loops to keep addressing simple and reads coalesced.
        for (int s = tx; s < 3 * (T + 1); s += blockDim.x) wf_M [s] = NEG_INF;
        for (int s = tx; s < 3 * (T + 1); s += blockDim.x) wf_IX[s] = NEG_INF;
        for (int s = tx; s < 3 * (T + 1); s += blockDim.x) wf_IY[s] = NEG_INF;

        __syncthreads();

        // --- Cooperative coalesced load of tile segments into shared memory ---
        // Mirrors the NW shared-memory load pattern exactly.
        for (int s = tx; s < refLen; s += blockDim.x)
            shared_ref[s] = d_seqs[refStart + reference_idx + s];
        for (int s = tx; s < qryLen; s += blockDim.x)
            shared_qry[s] = d_seqs[qryStart + query_idx + s];

        __syncthreads();

        // Per-thread local overlap-max accumulators (registers — no contention).
        // We track the best combined score across all three HMM states.
        int localMaxScore = NEG_INF;
        int localBest_i   = -1;
        int localBest_j   = -1;

        // =====================================================================
        // WAVEFRONT SCORING LOOP (anti-diagonal traversal)
        // Diagonal k: cells where i + j == k, i ∈ [max(0,k-qryLen), min(refLen,k)]
        // =====================================================================
        for (int k = 0; k <= refLen + qryLen; ++k) {

            // Cyclic 3-slot ring buffer indices (same scheme as NW)
            int curr_k   = (k % 3)       * (T + 1);
            int pre_k    = ((k + 2) % 3) * (T + 1);
            int prepre_k = ((k + 1) % 3) * (T + 1);

            int i_start = max(0, k - qryLen);
            int i_end   = min(refLen, k);

            // -----------------------------------------------------------------
            // WAVEFRONT PARALLELISM: each thread handles one (or more) cells
            // on this diagonal via striding.  Mirrors the NW parallel loop.
            // Max wavefront width = min(refLen, qryLen) ≤ T = 200, so with
            // BLOCK_SIZE = 256 a single pass usually covers the whole diagonal.
            // -----------------------------------------------------------------
            for (int i = i_start + tx; i <= i_end; i += blockDim.x) {
                int j = k - i;

                int16_t vm  = NEG_INF;
                int16_t vix = NEG_INF;
                int16_t viy = NEG_INF;
                uint8_t dir = DIR_DIAG;

                // -------------------------------------------------------------
                // Boundary / origin conditions
                // -------------------------------------------------------------
                if (i == 0 && j == 0) {
                    // Tile origin — seed from the previous tile's best overlap score.
                    vm  = tileStartScore;
                    vix = NEG_INF;
                    viy = NEG_INF;
                    dir = DIR_DIAG;
                }
                else if (i == 0) {
                    // Top edge — only IX (gap in ref) reachable; query advances left.
                    int16_t from_m  = (wf_M [pre_k + i] == NEG_INF) ? NEG_INF
                                      : wf_M [pre_k + i] + GAP_OPEN_M;
                    int16_t from_ix = (wf_IX[pre_k + i] == NEG_INF) ? NEG_INF
                                      : wf_IX[pre_k + i] + GAP_EXT;
                    vix = max(from_m, from_ix);
                    vm  = NEG_INF;
                    viy = NEG_INF;
                    dir = DIR_LEFT;
                }
                else if (j == 0) {
                    // Left edge — only IY (gap in query) reachable; ref advances up.
                    int16_t from_m  = (wf_M [pre_k + (i-1)] == NEG_INF) ? NEG_INF
                                      : wf_M [pre_k + (i-1)] + GAP_OPEN_M;
                    int16_t from_iy = (wf_IY[pre_k + (i-1)] == NEG_INF) ? NEG_INF
                                      : wf_IY[pre_k + (i-1)] + GAP_EXT;
                    viy = max(from_m, from_iy);
                    vm  = NEG_INF;
                    vix = NEG_INF;
                    dir = DIR_UP;
                }
                else {
                    // ---------------------------------------------------------
                    // Inner cell — full Pair HMM Viterbi recurrences
                    // Uses shared memory for r_char / q_char (coalesced access).
                    // ---------------------------------------------------------
                    char r_char = shared_ref[i - 1];
                    char q_char = shared_qry[j - 1];

                    int16_t emit_m = (r_char == q_char) ? MATCH : MISMATCH;

                    // ---- State M (diagonal predecessor): best of M, IX, IY → M ----
                    // VM[i][j] = emit_m + max( VM[i-1][j-1],
                    //                          VIX[i-1][j-1] + CLOSE_GAP,
                    //                          VIY[i-1][j-1] + CLOSE_GAP )
                    int16_t vm_from_m  = (wf_M [prepre_k + (i-1)] == NEG_INF) ? NEG_INF
                                         : wf_M [prepre_k + (i-1)] + emit_m;
                    int16_t vm_from_ix = (wf_IX[prepre_k + (i-1)] == NEG_INF) ? NEG_INF
                                         : wf_IX[prepre_k + (i-1)] + CLOSE_GAP + emit_m;
                    int16_t vm_from_iy = (wf_IY[prepre_k + (i-1)] == NEG_INF) ? NEG_INF
                                         : wf_IY[prepre_k + (i-1)] + CLOSE_GAP + emit_m;

                    // Three-way max — use DPX on sm_90+, otherwise plain comparisons.
                    // This mirrors the NW DPX bonus optimisation exactly.
                    #if __CUDA_ARCH__ >= 900
                        {
                            unsigned u = __vimax3_s16x2(
                                (unsigned)(vm_from_m  & 0xFFFF) | ((unsigned)(vm_from_m  & 0xFFFF) << 16),
                                (unsigned)(vm_from_ix & 0xFFFF) | ((unsigned)(vm_from_ix & 0xFFFF) << 16),
                                (unsigned)(vm_from_iy & 0xFFFF) | ((unsigned)(vm_from_iy & 0xFFFF) << 16));
                            vm = (int16_t)(u & 0xFFFF);
                        }
                    #else
                        vm = vm_from_m;
                        if (vm_from_ix > vm) vm = vm_from_ix;
                        if (vm_from_iy > vm) vm = vm_from_iy;
                    #endif

                    // Traceback direction for M: which predecessor won?
                    dir = DIR_DIAG;
                    if (vm_from_ix > vm_from_m  && vm_from_ix >= vm_from_iy) dir = DIR_LEFT;
                    if (vm_from_iy > vm_from_m  && vm_from_iy >  vm_from_ix) dir = DIR_UP;

                    // ---- State IX (gap in ref / left): query advances ----
                    // VIX[i][j] = max( VM [i][j-1] + GAP_OPEN_M,
                    //                  VIX[i][j-1] + GAP_EXT,
                    //                  VIY[i][j-1] + SWITCH_GAP )
                    int16_t vix_m  = (wf_M [pre_k + i] == NEG_INF) ? NEG_INF
                                     : wf_M [pre_k + i] + GAP_OPEN_M;
                    int16_t vix_ix = (wf_IX[pre_k + i] == NEG_INF) ? NEG_INF
                                     : wf_IX[pre_k + i] + GAP_EXT;
                    int16_t vix_iy = (wf_IY[pre_k + i] == NEG_INF) ? NEG_INF
                                     : wf_IY[pre_k + i] + SWITCH_GAP;
                    #if __CUDA_ARCH__ >= 900
                        {
                            unsigned u = __vimax3_s16x2(
                                (unsigned)(vix_m  & 0xFFFF) | ((unsigned)(vix_m  & 0xFFFF) << 16),
                                (unsigned)(vix_ix & 0xFFFF) | ((unsigned)(vix_ix & 0xFFFF) << 16),
                                (unsigned)(vix_iy & 0xFFFF) | ((unsigned)(vix_iy & 0xFFFF) << 16));
                            vix = (int16_t)(u & 0xFFFF);
                        }
                    #else
                        vix = max(vix_m, max(vix_ix, vix_iy));
                    #endif

                    // ---- State IY (gap in query / up): ref advances ----
                    // VIY[i][j] = max( VM [i-1][j] + GAP_OPEN_M,
                    //                  VIX[i-1][j] + SWITCH_GAP,
                    //                  VIY[i-1][j] + GAP_EXT )
                    int16_t viy_m  = (wf_M [pre_k + (i-1)] == NEG_INF) ? NEG_INF
                                     : wf_M [pre_k + (i-1)] + GAP_OPEN_M;
                    int16_t viy_ix = (wf_IX[pre_k + (i-1)] == NEG_INF) ? NEG_INF
                                     : wf_IX[pre_k + (i-1)] + SWITCH_GAP;
                    int16_t viy_iy = (wf_IY[pre_k + (i-1)] == NEG_INF) ? NEG_INF
                                     : wf_IY[pre_k + (i-1)] + GAP_EXT;
                    #if __CUDA_ARCH__ >= 900
                        {
                            unsigned u = __vimax3_s16x2(
                                (unsigned)(viy_m  & 0xFFFF) | ((unsigned)(viy_m  & 0xFFFF) << 16),
                                (unsigned)(viy_ix & 0xFFFF) | ((unsigned)(viy_ix & 0xFFFF) << 16),
                                (unsigned)(viy_iy & 0xFFFF) | ((unsigned)(viy_iy & 0xFFFF) << 16));
                            viy = (int16_t)(u & 0xFFFF);
                        }
                    #else
                        viy = max(viy_m, max(viy_ix, viy_iy));
                    #endif
                }

                // --- Write computed wavefront values for all three states ---
                wf_M [curr_k + i] = vm;
                wf_IX[curr_k + i] = vix;
                wf_IY[curr_k + i] = viy;

                // --- Store traceback direction for inner cells ---
                // The overall best state at (i,j) determines the pointer.
                // If IX or IY beats M, override the direction accordingly.
                if (i > 0 && j > 0) {
                    int16_t cell_best = vm;
                    uint8_t cell_dir  = dir;           // direction from M state
                    if (vix > cell_best) { cell_best = vix; cell_dir = DIR_LEFT; }
                    if (viy > cell_best) { cell_best = viy; cell_dir = DIR_UP;   }
                    tbDir[(i - 1) * T + (j - 1)] = cell_dir;
                }

                // --- GACT overlap tracking (per-thread local accumulator) ---
                // Track best score in the overlap region across all HMM states.
                // Parallel reduction happens after the wavefront loop ends.
                // This mirrors the NW localMaxScore accumulation pattern.
                if (!lastTile) {
                    if (i > (refLen - O) && j > (qryLen - O)) {
                        int16_t cell_best = max(vm, max(vix, viy));
                        if ((int)cell_best > localMaxScore) {
                            localMaxScore = (int)cell_best;
                            localBest_i   = i;
                            localBest_j   = j;
                        }
                    }
                }

            } // end strided wavefront cell loop

            // All threads must finish writing wf_M/IX/IY before the next diagonal
            // reads them.  This is the critical synchronisation point — mirrors NW.
            __syncthreads();

        } // end wavefront (diagonal) loop

        // =====================================================================
        // PARALLEL REDUCTION — find best overlap (i, j, score)
        // Identical tree-reduction structure to the NW kernel.
        // =====================================================================
        if (tx < BLOCK_SIZE) {
            s_maxScore[tx] = localMaxScore;
            s_best_i  [tx] = localBest_i;
            s_best_j  [tx] = localBest_j;
        }
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tx < stride) {
                if (s_maxScore[tx + stride] > s_maxScore[tx]) {
                    s_maxScore[tx] = s_maxScore[tx + stride];
                    s_best_i  [tx] = s_best_i  [tx + stride];
                    s_best_j  [tx] = s_best_j  [tx + stride];
                }
            }
            __syncthreads();
        }
        // s_maxScore[0], s_best_i[0], s_best_j[0] now hold the global tile maximum.

        if (tx == 0 && !lastTile) {
            sh_next_ref_advance = s_best_i[0];
            sh_next_qry_advance = s_best_j[0];
        }
        __syncthreads();

        // =====================================================================
        // TRACEBACK (single thread — inherently sequential)
        // Mirrors the NW traceback: thread 0 walks backwards through tbDir,
        // writing into shared localPath.
        // =====================================================================
        if (tx == 0) {
            sh_localLen = 0;

            int ti = (!lastTile) ? sh_next_ref_advance : (int)refLen;
            int tj = (!lastTile) ? sh_next_qry_advance : (int)qryLen;

            // Carry the best overlap score into the next tile's origin cell.
            tileStartScore = (int16_t)s_maxScore[0];

            // Also lock in the advance amounts for use after traceback.
            sh_next_ref_advance = ti;
            sh_next_qry_advance = tj;

            while (ti > 0 || tj > 0) {
                uint8_t d;

                if      (ti == 0) d = DIR_LEFT;
                else if (tj == 0) d = DIR_UP;
                else              d = tbDir[(ti - 1) * T + (tj - 1)];

                localPath[sh_localLen++] = d;

                if      (d == DIR_DIAG) { ti--; tj--; }
                else if (d == DIR_UP)   { ti--;       }
                else                    {       tj--; }
            }
        }
        __syncthreads();

        // =====================================================================
        // COALESCED GLOBAL WRITE — reverse localPath into d_tb
        // Threads cooperate to write in forward order with stride blockDim.x,
        // matching the NW coalesced write optimisation.
        // =====================================================================
        int   localLen         = sh_localLen;
        int32_t currentPathLen = sh_currentPairPathLen;

        for (int s = tx; s < localLen; s += blockDim.x) {
            d_tb[tbGlobalOffset + currentPathLen + s] = localPath[localLen - 1 - s];
        }
        __syncthreads();

        // =====================================================================
        // ADVANCE TILE CURSORS — thread 0 updates shared state
        // =====================================================================
        if (tx == 0) {
            sh_currentPairPathLen += sh_localLen;
            sh_reference_idx      += sh_next_ref_advance;
            sh_query_idx          += sh_next_qry_advance;
        }
        __syncthreads();
        // The __syncthreads() above ensures all threads see the updated cursors
        // before re-evaluating the while(!lastTile) condition.

    } // end tile loop
    // No inter-pair sync needed — each block handles exactly one pair.
}


// ============================================================================
// Host: Reconstruct aligned strings from traceback paths
// ============================================================================
void GpuAligner::getAlignedSequences(TB_PATH& tb_paths) {

    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    int tb_length = longestLen << 1;

    // CPU-side parallelism with OpenMP (mirrors NW host code)
    #pragma omp parallel for
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
                // Match / Mismatch — both sequences advance
                aln0 += seq0[seqPos0];
                aln1 += seq1[seqPos1];
                seqPos0++; seqPos1++;
            }
            else if (tb_paths[i] == DIR_UP) {
                // IY state — ref advances, query gets gap
                aln0 += seq0[seqPos0];
                aln1 += '-';
                seqPos0++;
            }
            else if (tb_paths[i] == DIR_LEFT) {
                // IX state — query advances, ref gets gap
                aln0 += '-';
                aln1 += seq1[seqPos1];
                seqPos1++;
            }
            else {
                break;  // end-of-path sentinel (0 / uninitialised)
            }
        }

        seqs[seqId0].aln = aln0;
        seqs[seqId1].aln = aln1;
    }
}


// ============================================================================
// Cleanup
// ============================================================================
void GpuAligner::clearAndReset() {
    cudaFree(d_seqs);
    cudaFree(d_seqLen);
    cudaFree(d_tb);
    seqs.clear();
    longestLen = 0;
    numPairs   = 0;
}


// ============================================================================
// Main orchestration
// ============================================================================
void GpuAligner::alignment() {

    // One block per pair; 256 threads per block.
    // NOTE: BLOCK_SIZE constant inside the kernel must match blockSize here.
    // 256 threads ≥ T (= 200), so the widest diagonal is always covered in a
    // single strided pass, exactly as in the NW parallel kernel.
    int numBlocks = numPairs;
    int blockSize = 256;

    allocateMem();
    transferSequence2Device();

    alignmentOnGPU<<<numBlocks, blockSize>>>(d_info, d_seqLen, d_seqs, d_tb);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    TB_PATH tb_paths = transferTB2Host();
    cudaDeviceSynchronize();

    getAlignedSequences(tb_paths);
}


// ============================================================================
// Output
// ============================================================================
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
        outFile << (seq.aln  + '\n');
    }
    outFile.close();
}