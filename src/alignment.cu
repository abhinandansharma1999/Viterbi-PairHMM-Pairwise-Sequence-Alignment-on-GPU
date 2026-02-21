#include "alignment.cuh"
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <fstream>

#include <cuda_runtime.h>
#include <cuda.h>

#if __CUDA_ARCH__ >= 900
#include <cuda/dpx_intrinsics.h>
#endif

/**
 * Allocates GPU memory for sequences, lengths, and traceback paths.
 * Calculates 'longestLen' to determine the stride for flattening the sequence array.
 */
void GpuAligner::allocateMem() {
    // 1. Find the maximum sequence length to determine memory stride
    longestLen = std::max_element(seqs.begin(), seqs.end(), [](const Sequence& a, const Sequence& b) {
        return a.seq.size() < b.seq.size();
    })->seq.size();

    cudaError_t err;
    
    // 2. Allocate flat array for all sequences (Reference + Query pairs)
    // Layout: [Seq0_Ref ...pad... | Seq0_Qry ...pad... | Seq1_Ref ... ]
    err = cudaMalloc(&d_seqs, numPairs * 2 * longestLen * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 3. Allocate array for sequence lengths (to handle padding correctly)
    err = cudaMalloc(&d_seqLen, numPairs * 2 * sizeof(int32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 4. Allocate Traceback Buffer
    // Worst case path is roughly 2x sequence length (all gaps)
    int tb_length = longestLen << 1; 
    err = cudaMalloc(&d_tb, numPairs * tb_length * sizeof(uint8_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 5. Allocate meta-info struct (numPairs, maxLen)
    err = cudaMalloc(&d_info, 2 * sizeof(int32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
}

/**
 * Flattens the host sequence objects into a single 1D array and transfers to GPU.
 */
void GpuAligner::transferSequence2Device() {
    cudaError_t err;
    
    // 1. Flatten sequences on Host
    // We use a fixed stride 'longestLen' to simplify indexing on the GPU
    std::vector<char> h_seqs(longestLen * numPairs * 2, 0); 
    
    for (size_t i = 0; i < numPairs * 2; ++i) {
        const std::string& s = seqs[i].seq;
        std::memcpy(h_seqs.data() + (i * longestLen), s.data(), s.size());
    }

    // 2. Transfer flattened sequences to Device
    err = cudaMemcpy(d_seqs, h_seqs.data(), longestLen * numPairs * 2 * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 3. Transfer sequence lengths to Device
    std::vector<int32_t> h_seqLen(numPairs * 2, 0);
    for (int i = 0; i < numPairs * 2; ++i) h_seqLen[i] = seqs[i].seq.size();
    
    err = cudaMemcpy(d_seqLen, h_seqLen.data(), numPairs * 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 4. Initialize Traceback buffer on Device (Zero out)
    int tb_length = longestLen << 1;
    std::vector<uint8_t> h_tb (tb_length * numPairs, 0);
    
    err = cudaMemcpy(d_tb, h_tb.data(), tb_length * numPairs * sizeof(uint8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }

    // 5. Transfer Meta Info
    std::vector<int32_t> h_info (2);
    h_info[0] = numPairs;
    h_info[1] = longestLen;
    err = cudaMemcpy(d_info, h_info.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
}

/**
 * Copies the computed traceback paths from GPU back to Host.
 */
TB_PATH GpuAligner::transferTB2Host() {
    int tb_length = longestLen << 1;
    TB_PATH h_tb(tb_length * numPairs);

    cudaError_t err = cudaMemcpy(h_tb.data(), d_tb, tb_length * numPairs * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
    return h_tb;
}

/**
 * CUDA Kernel: Performs tiled alignment (GACT) on the GPU.
 * TODO: Optimize using shared memory, wavefront parallelism, and memory coalescing.
 * HINT: 
 * Consider
 * 1. Number of threads for each step (initialization, filling scoring matrix, traceback)
 * 2. Which variables should go in registers vs shared memory
 * 3. Where should __syncthreads() be added?
 * 4. TODOs marked below are the main tasks, but other parts may also need changes for correctness or efficiency
 * 5. You may add/modify shared memory, registers, or helper functions as needed, as long as output is valid
 */
__global__ void alignmentOnGPU (
    int32_t* d_info,       // [0]: numPairs, [1]: maxSeqLen
    int32_t* d_seqLen,     // Array of sequence lengths
    char* d_seqs,          // Flat array of sequences
    uint8_t* d_tb          // Output traceback paths
) {
    // -----------------------------------------------------------
    // KERNEL CONFIGURATION
    // -----------------------------------------------------------
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    
    // GACT Parameters
    // TODO: Adjust the tile size and overlap region to explore the tradeoff between speed and accuracy
    const int T = 200;        // Tile size
    const int O = 64;         // Overlap between tiles

    const int BLOCK_SIZE = 256;
    
    // Scoring Scheme (DO NOT MODIFY)
    const int16_t MATCH = 2;
    const int16_t MISMATCH = -1;
    const int16_t GAP = -2;

    // Traceback Direction Constants (DO NOT MODIFY)
    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;

    // Shared Memory Allocation
    // tbDir: Stores direction for every cell in the tile (T x T)
    // Note: We only store directions for the inner matrix (indices 1..T)
    __shared__ uint8_t tbDir[T * T]; 
    
    // wf_scores: 3 arrays (Current, Previous, Pre-Previous wavefronts) 
    // needed for diagonal computation. Size T+1 to include boundary 0.
    __shared__ int16_t wf_scores[3 * (T + 1)]; 

    // localPath: Temporary buffer to store tile traceback (reversed)
    __shared__ uint8_t localPath[2 * T];

    // State variables
    __shared__ bool lastTile;
    __shared__ int16_t tileStartScore;  // carries best overlap score into next tile's origin cell

    __shared__ int s_maxScore[BLOCK_SIZE];
    __shared__ int s_best_i[BLOCK_SIZE];
    __shared__ int s_best_j[BLOCK_SIZE];

    // HINT: Use shared memory to store the reference and query segments involved in the tile
    __shared__ char shared_ref[T];
    __shared__ char shared_qry[T];

    // Shared state needed across threads within a tile
    __shared__ int32_t sh_localLen;
    __shared__ int32_t sh_currentPairPathLen;
    __shared__ int32_t sh_reference_idx;
    __shared__ int32_t sh_query_idx;
    __shared__ int32_t sh_next_ref_advance;
    __shared__ int32_t sh_next_qry_advance;

    // int32_t numPairs = d_info[0];
    int32_t maxSeqLen = d_info[1]; 

    // Iterate over every pair of sequences
    // TODO: Parallelize – assign one block per alignment pair
    int pair = bx;
    
    // --- Initialization per Pair ---
    // HINT:
    // Think carefully,
    // 1. Should these variables be stored in registers or shared memory?
    //    (feel free to modify the baseline code)
    // 2. How many threads are needed to initialize shared memory efficiently?
    // 3. Should __syncthreads() be added after initialization?

    if (tx == 0) {
        lastTile = false;
        tileStartScore = 0;
        sh_currentPairPathLen = 0;
        sh_reference_idx = 0;
        sh_query_idx = 0;
    }
    __syncthreads();

    // Calculate memory offsets for this pair
    int32_t refStart = (pair * 2) * maxSeqLen;
    int32_t qryStart = (pair * 2 + 1) * maxSeqLen;
    int32_t tbGlobalOffset = pair * (maxSeqLen * 2); 
    
    int32_t refTotalLen = d_seqLen[2 * pair];
    int32_t qryTotalLen = d_seqLen[2 * pair + 1];
       
    
    // -------------------------------------------------------
    // TILE LOOP: Align the sequence tile by tile
    // -------------------------------------------------------
    while (!lastTile) {
        
        // Register-local copies of shared indices for this tile iteration
        int32_t reference_idx = sh_reference_idx;
        int32_t query_idx     = sh_query_idx;

        // Determine Tile Size (Clip to sequence end)
        int32_t refLen = min(T, refTotalLen - reference_idx); 
        int32_t qryLen = min(T, qryTotalLen - query_idx); 
        
        // Check termination conditions
        // HINT:
        // 1. How many threads are needed here?
        if (tx == 0) {
            if ((reference_idx + refLen == refTotalLen) && (query_idx + qryLen == qryTotalLen)) lastTile = true;
        }

        // Reset Wavefront Scores (approx -infinity)
        // HINT: 
        // 1. How many threads are needed to efficiently initialize wf_scores?
        // 2. Is the initialization really necessary, or can it be omitted?
        for (int s = tx; s < 3 * (T + 1); s += blockDim.x) wf_scores[s] = -9999;
        
        __syncthreads();

        // TODO: Load the reference and query segments from global memory into shared memory
        // HINT: Using memory coalescing
        for (int s = tx; s < refLen; s += blockDim.x) shared_ref[s] = d_seqs[refStart + reference_idx + s];
        for (int s = tx; s < qryLen; s += blockDim.x) shared_qry[s] = d_seqs[qryStart + query_idx + s];

        __syncthreads();

        // Per-thread local variables for tracking overlap max score
        // during the wavefront loop. Accumulated into registers (fast, no contention),
        // then reduced into shared memory after the wavefront loop completes.
        int localMaxScore = -9999;
        int localBest_i = -1;
        int localBest_j = -1;
        

        // ---------------------------------------------------
        // WAVEFRONT SCORING LOOP (Diagonal Traversal)
        // ---------------------------------------------------
        for (int k = 0; k <= refLen + qryLen; ++k) {
            
            // Cyclic buffers for 3-wavefront dependency
            int curr_k   = (k % 3) * (T + 1);
            int pre_k    = ((k + 2) % 3) * (T + 1);
            int prepre_k = ((k + 1) % 3) * (T + 1);

            // Compute loop bounds for this diagonal
            int i_start = max(0, k - qryLen);
            int i_end   = min(refLen, k);

            // TODO: Implement wavefront parallelism
            // HINT: 
            // 1. Assign each thread to a cell on the wavefront
            // 2. What is the maximum possible length of wavefront? 
            //    Can we always set blockSize to avoid situations 
            //    where the wavefront size exceeds the number of available block threads?
            for (int i = i_start + tx; i <= i_end; i += blockDim.x) {
                int j = k - i; 
                
                int16_t score = -9999;
                uint8_t direction = DIR_DIAG;

                // -- Boundary Conditions --
                if (i == 0 && j == 0) {
                    score = tileStartScore; // Score from the previous tile
                } 
                else if (i == 0) {
                    score = wf_scores[pre_k + i] + GAP; // Gap from Left
                    direction = DIR_LEFT;
                } 
                else if (j == 0) {
                    score = wf_scores[pre_k + (i - 1)] + GAP; // Gap from Up
                    direction = DIR_UP;
                } 
                else {
                    // -- Inner Matrix Calculation --
                    // TODO: Replace this global memory access with shared memory
                    char r_char = shared_ref[i - 1];
                    char q_char = shared_qry[j - 1];
                    
                    int16_t score_diag = wf_scores[prepre_k + (i - 1)] + (r_char == q_char ? MATCH : MISMATCH);
                    int16_t score_up   = wf_scores[pre_k + (i - 1)] + GAP;
                    int16_t score_left = wf_scores[pre_k + i] + GAP;
                
                    // Find Max (BONUS: Replace with DPX instructions)
                    #if __CUDA_ARCH__ >= 900
                        // DPX accelerated max3
                        int best = __vimax3_s16x2(
                            (score_diag & 0xFFFF) | (score_diag << 16),
                            (score_up   & 0xFFFF) | (score_up   << 16),
                            (score_left & 0xFFFF) | (score_left << 16)
                        );

                        score = (int16_t)(best & 0xFFFF);
                    #else
                        score = score_diag;
                        if (score_up > score)   score = score_up;
                        if (score_left > score) score = score_left;
                    #endif

                    // Direction (still needed for traceback)
                    if (score == score_diag)      direction = DIR_DIAG;
                    else if (score == score_up)   direction = DIR_UP;
                    else                           direction = DIR_LEFT;
                }

                // Write Score
                wf_scores[curr_k + i] = score;
                
                // Write Direction (Only for inner cells, shifted index)
                if (i > 0 && j > 0) {
                        tbDir[(i - 1) * T + (j - 1)] = direction;
                }

                // -- GACT Overlap Logic --
                // Track the highest score in the overlap region (past T-O)
                // TODO: Compute the max score and its indices for cells in the overlap region
                // Sequentially, max score can be updated during iteration.
                // For parallel execution, think about how to handle this safely.
                // HINT: Parallel Reduction

                if (!lastTile) {
                    if (i > (refLen - O) && j > (qryLen - O)) {
                        if (score > localMaxScore) {
                            localMaxScore = score;
                            localBest_i = i;
                            localBest_j = j;
                        }
                    }
                }                    
            }

            __syncthreads();
        } // End Wavefront Loop

        if (tx < BLOCK_SIZE) {
            s_maxScore[tx] = localMaxScore;
            s_best_i[tx]   = localBest_i;
            s_best_j[tx]   = localBest_j;
        }
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tx < stride) {
                if (s_maxScore[tx + stride] > s_maxScore[tx]) {
                    s_maxScore[tx] = s_maxScore[tx + stride];
                    s_best_i[tx]   = s_best_i[tx + stride];
                    s_best_j[tx]   = s_best_j[tx + stride];
                }
            }
            __syncthreads();
        }

        // tx==0 writes the reduced result to shared state for use in traceback
        if (tx == 0 && !lastTile) {
            sh_next_ref_advance = s_best_i[0];
            sh_next_qry_advance = s_best_j[0];
        }
        __syncthreads();

        // ---------------------------------------------------
        // TRACEBACK & REVERSAL
        // ---------------------------------------------------
        // HINT:
        // 1. How many threads should be used for the traceback?
        // 2. Consider carefully whether the variable should be stored in registers or shared memory.

        if (tx == 0) {
            sh_localLen = 0;

            // Determine where to start traceback (Overlap heuristic vs End of Tile)
            int ti = (!lastTile) ? sh_next_ref_advance : refLen;
            int tj = (!lastTile) ? sh_next_qry_advance : qryLen;

            // Carry the best overlap score into the next tile's origin cell
            tileStartScore = (int16_t)s_maxScore[0];

            // Re-store the actual advance amounts (ti/tj may equal refLen/qryLen on lastTile)
            sh_next_ref_advance = ti;
            sh_next_qry_advance = tj;

            // Traceback Backwards (End -> Start) into Shared Memory
            while (ti > 0 || tj > 0) {
                uint8_t dir;

                // Implicit boundary handling for top/left edges
                if (ti == 0) { 
                    dir = DIR_LEFT; 
                } else if (tj == 0) { 
                    dir = DIR_UP;   
                } else {
                    // Fetch direction from DP table
                    dir = tbDir[(ti - 1) * T + (tj - 1)];
                }

                // Store to local temporary buffer
                localPath[sh_localLen] = dir;
                sh_localLen++;
                
                // Move coordinates
                if (dir == DIR_DIAG) { ti--; tj--; } 
                else if (dir == DIR_UP) { ti--; } 
                else { tj--; }
            }
        }

        __syncthreads();

        // Write Forward (Start -> End) to Global Memory
        // Reverses the local path so the CPU gets it in forward order
        // TODO: Use memory coalescing to efficiently write the data to global memory
        int localLen = sh_localLen;
        int32_t currentPairPathLen = sh_currentPairPathLen;
        for (int k = tx; k < localLen; k += blockDim.x) {
            d_tb[tbGlobalOffset + currentPairPathLen + k] = localPath[localLen - 1 - k];
        }

        __syncthreads();
        
        // ---------------------------------------------------
        // ADVANCE TO NEXT TILE
        // ---------------------------------------------------
        // Only thread 0 updates shared cursor variables so there
        // is no race; all threads will see the updated values after
        // __syncthreads() at the top of the next tile iteration
        if (tx == 0) {
            sh_currentPairPathLen += sh_localLen;
            sh_reference_idx += sh_next_ref_advance;
            sh_query_idx     += sh_next_qry_advance;
        }
        __syncthreads();
        // HINT: Is __syncthreads() needed after each tile?
        // Yes – the __syncthreads() above ensures sh_reference_idx/sh_query_idx
        // are visible to all threads before the next tile's while-condition check.

    } // End Tile Loop

    // HINT: Is __syncthreads() needed after each alignment?
    // No – each block handles exactly one pair, so no inter-pair sync is needed.
}

/** * Reconstructs the actual string alignment from the traceback paths (CIGAR-like data).
 * Converts directional codes (DIAG, UP, LEFT) into aligned strings with gaps.
 */
void GpuAligner::getAlignedSequences (TB_PATH& tb_paths) {

    const uint8_t DIR_DIAG = 1;
    const uint8_t DIR_UP   = 2;
    const uint8_t DIR_LEFT = 3;
    
    int tb_length = longestLen << 1;
    
    // TODO: Apply parallelism to this for loop
    // HINT: Remember to add the header
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

        // Iterate through the recorded path directions
        for (int i = tb_start; i < tb_start+tb_length; ++i) {
            if (tb_paths[i] == DIR_DIAG) {
                // Match/Mismatch
                aln0 += seq0[seqPos0];
                aln1 += seq1[seqPos1];
                seqPos0++; seqPos1++;
            }
            else if (tb_paths[i] == DIR_UP) {
                // Deletions (gap on seq1)
                aln0 += seq0[seqPos0];
                aln1 += '-';
                seqPos0++;
            }
            else if (tb_paths[i] == DIR_LEFT) {
                // Insertions (gap on seq0)
                aln0 += '-';
                aln1 += seq1[seqPos1];
                seqPos1++;
            }
            else {
                // End of the tb_path (encountered 0 or uninitialized value)
                break;
            }
        }

        // Save results
        seqs[seqId0].aln = aln0;
        seqs[seqId1].aln = aln1;
    }
}

void GpuAligner::clearAndReset () {
    cudaFree(d_seqs);
    cudaFree(d_seqLen);
    cudaFree(d_tb);
    seqs.clear();
    longestLen = 0;
    numPairs = 0;
}

/**
 * Main orchestration method.
 * 1. Allocates GPU memory
 * 2. Transfers data
 * 3. Launches Kernel
 * 4. Retrieves results and reconstructs alignment strings
 */
void GpuAligner::alignment () {

    // TODO: make sure to appropriately set the values below
    int numBlocks = numPairs;  // i.e. number of thread blocks on the GPU
    // printf("%d\n", numBlocks);
    int blockSize = 256; // i.e. number of GPU threads per thread block
    // NOTE: blockSize must match BLOCK_SIZE constant defined in the kernel above

    // 1. Allocate memory on Device
    allocateMem();
    
    // 2. Transfer sequence to device
    transferSequence2Device();
    
    // 3. Perform the alignment on GPU
    alignmentOnGPU<<<numBlocks, blockSize>>>(d_info, d_seqLen, d_seqs, d_tb);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: %s (%s)\n", cudaGetErrorString(err), cudaGetErrorName(err));
        exit(1);
    }
    
    // 4. Transfer the traceback path from device
    TB_PATH tb_paths = transferTB2Host();
    cudaDeviceSynchronize();
    
    // 5. Get the aligned sequence with traceback paths
    getAlignedSequences(tb_paths);
    
}

/**
 * Writes the aligned sequences to a file in FASTA format.
 * Each sequence is written with a header line ('>' + name) followed by the aligned sequence.
 * If `append` is true, the output is appended to the file; otherwise, the file is overwritten.
 */
void GpuAligner::writeAlignment(std::string fileName, bool append) {
    std::ofstream outFile;
    if (append) outFile.open(fileName, std::ios::app);
    else        outFile.open(fileName);
    if (!outFile) {
        fprintf(stderr, "ERROR: cant open file: %s\n", fileName.c_str());
        exit(1);
    }
    for (auto& seq: seqs) {
        outFile << ('>' + seq.name + '\n');
        outFile << (seq.aln + '\n');
    }
    outFile.close();
}
