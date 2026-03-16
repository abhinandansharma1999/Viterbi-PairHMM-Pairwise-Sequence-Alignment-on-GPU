# GPU-Accelerated Pairwise Sequence Alignment — Run Guide

Pair-HMM Viterbi alignment with GACT tiling, implemented across CPU and multiple GPU optimization levels.

---

## Repository Structure

```
FINAL_PROJECT/
├── data/                              # Input FASTA files and reference alignments (download separately)
│   ├── sequences.fa                   # Dataset 1: 5,000 pairs, 200–10,000 bp
│   ├── reference_alignment.fa         # Dataset 1 reference — linear gap penalties
│   ├── hmm_fullDP_ref.fa              # Dataset 1 reference — affine gap penalties (Pair-HMM full DP)
│   ├── sars_20000.fa                  # Dataset 2: 1,000 SARS-CoV-2 pairs
│   └── sars_20000_ref.fa              # Dataset 2 reference — affine gap penalties
├── src/
│   ├── main.cpp                       # Entry point — edit include and printGpuProperties() per mode
│   ├── alignment.cpp                  # CPU implementation file (copy from submission-files/)
│   ├── alignment.cu                   # GPU implementation file (copy from submission-files/)
│   ├── alignment.cuh                  # GPU header (do not modify)
│   ├── alignment.h                    # CPU header (do not modify)
│   ├── check_alignment.cpp            # Tool: verifies aligned output reproduces original sequences
│   ├── compare_alignment.cpp          # Tool: compares alignment scores vs. reference
│   ├── findgpu.cu                     # CMake helper: detects GPU architecture
│   ├── gpuProperty.cu                 # Prints GPU device properties
│   ├── kseq.h                         # FASTA/FASTQ parser
│   └── timer.hpp                      # Performance timer
├── submission-files/                  # All implementation variants (see details below)
│   ├── alignment_pairhmm_full_DP_w_progress.cpp
│   ├── alignment_CPU_execution.cpp
│   ├── alignment_hmm_gpu_1_thread_same_loss.cu
│   ├── alignment_hmm_gpu_1_thread_same_loss.cuh
│   ├── alignment_hmm_gpu_level_1.cu
│   ├── alignment_hmm_gpu_level_2.cu
│   ├── alignment_hmm_gpu_level_3.cu
│   └── generate_plot.py
├── CMakeLists.txt                     # Build system — edit add_executable per mode
└── run-commands.sh                    # Build and run script — uncomment dataset commands as needed
```

---

## Datasets

Download the FASTA files from Google Drive and place them in the `data/` folder.
Drive Link: https://drive.google.com/drive/folders/1vuBk0X0oZQkze6p5PV1LSBExI6B1jii0?usp=sharing

| File | Description |
|------|-------------|
| `sequences.fa` | Dataset 1 — 5,000 sequence pairs, lengths 200–10,000 bp |
| `reference_alignment.fa` | Dataset 1 reference alignment using linear gap penalties |
| `hmm_fullDP_ref.fa` | Dataset 1 reference alignment using affine gap penalties (Pair-HMM full DP) |
| `sars_20000.fa` | Dataset 2 — 20,000 SARS-CoV-2 sequence pairs (we only generated reference for 1000 pairs) |
| `sars_20000_ref.fa` | Dataset 2 reference alignment using affine gap penalties |

---

## Implementation Modes

There are five implementation variants, each requiring specific file changes before building. The steps for each are described below.

---

### Mode 1 — CPU Full DP (Reference Generator)

Runs the complete O(mn) Viterbi DP matrix for each pair with no GACT tiling. Used to generate the gold-standard reference alignments (`hmm_fullDP_ref.fa`, `sars_20000_ref.fa`). Runtime is ~1 hour for Dataset 1 and ~2 hours for Dataset 2.

**Step 1 — Copy the implementation:**
```bash
cp submission-files/alignment_pairhmm_full_DP_w_progress.cpp src/alignment.cpp
```

**Step 2 — Edit `src/main.cpp`:**
- Change the include at the top from `alignment.cuh` → `alignment.h`
- Comment out the `printGpuProperties();` function call

```cpp
// #include "alignment.cuh"
#include "alignment.h"
// ...
// printGpuProperties();   // comment this out
```

**Step 3 — Edit `CMakeLists.txt`**, set the `add_executable` block for `aligner` to:
```cmake
add_executable(aligner
    src/main.cpp
    src/alignment.cpp
)
```

**Step 4 — Edit `run-commands.sh`**, uncomment the appropriate dataset command.

---

### Mode 2 — CPU GACT (Tiled CPU Execution)

Runs Pair-HMM Viterbi with GACT tiling (T=200, O=64) on CPU. Much faster than full DP (~4,000 ms for Dataset 1) with only 0.056% score loss.

**Step 1 — Copy the implementation:**
```bash
cp submission-files/alignment_CPU_execution.cpp src/alignment.cpp
```

**Step 2 — Edit `src/main.cpp`:** same as Mode 1 — use `alignment.h` and comment out `printGpuProperties()`.

**Step 3 — Edit `CMakeLists.txt`:** same as Mode 1.

**Step 4 — Edit `run-commands.sh`**, uncomment the appropriate dataset command.

---

### Mode 3 — GPU Single Thread (Correctness Baseline)

Runs the entire alignment on a single GPU thread. Functionally identical to the CPU GACT version but executes on the GPU with all data in global memory. This exists as a correctness stepping-stone — it is ~50x slower than CPU GACT due to GPU memory latency and zero parallelism.

**Step 1 — Copy the implementation:**
```bash
cp submission-files/alignment_hmm_gpu_1_thread_same_loss.cu src/alignment.cu
```

**Step 2 — Edit `src/main.cpp`:** use `alignment.cuh` and keep `printGpuProperties()` enabled (default state).

**Step 3 — Edit `CMakeLists.txt`**, set the `add_executable` block for `aligner` to:
```cmake
add_executable(aligner
    src/main.cpp
    src/alignment.cu
    src/gpuProperty.cu
)
```

**Step 4 — Edit `run-commands.sh`**, uncomment the appropriate dataset command.

---

### Mode 4 — GPU Level 1 (One Block Per Pair)

Each sequence pair is assigned its own CUDA block (1 thread/block). All pairs run in parallel across the GPU's SMs. ~2x speedup over CPU GACT (~43,716 ms for Dataset 1).

**Step 1 — Copy the implementation:**
```bash
cp submission-files/alignment_hmm_gpu_level_1.cu src/alignment.cu
```

**Step 2–4:** Same as Mode 3 (use `alignment.cuh`, GPU `CMakeLists.txt`, uncomment run command).

---

### Mode 5 — GPU Level 2 (Wavefront Parallelism)

Extends Level 1 by parallelising the anti-diagonal cell loop across 256 threads per block using `__syncthreads()` barriers. Wavefront arrays moved to shared memory. ~4x speedup over Level 1 (~10,844 ms for Dataset 1).

**Step 1 — Copy the implementation:**
```bash
cp submission-files/alignment_hmm_gpu_level_2.cu src/alignment.cu
```

**Step 2–4:** Same as Mode 3.

---

### Mode 6 — GPU Level 3 (Final Optimized — Reduced Global Memory Writes)

Extends Level 2 by eliminating redundant `tbDir` global memory writes for inner cells (direction is derived from HMM state directly during traceback). ~2.6x further speedup over Level 2 (~4,200 ms for Dataset 1). **This is the final, best-performing implementation.**

**Step 1 — Copy the implementation:**
```bash
cp submission-files/alignment_hmm_gpu_level_3.cu src/alignment.cu
```

**Step 2–4:** Same as Mode 3.

---

## Building and Running

All modes use the same build and run script:

```bash
bash run-commands.sh
```

The script will:
1. Create and enter the `build/` directory
2. Run `cmake ..` and `make -j4`
3. Set `LD_LIBRARY_PATH` for TBB
4. Execute the `aligner` binary with the dataset command you uncommented

**Key `aligner` flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `-i / --sequence` | — | Input FASTA file (required) |
| `-o / --output` | — | Output alignment FASTA file (required) |
| `-N / --maxPairs` | 5000 | Maximum number of sequence pairs to process (1–5000) |
| `-b / --batchSize` | 1000 | Pairs per GPU batch |
| `-T / --numThreads` | 4 | CPU threads for TBB (1–8) |

---

## Validating Output

After running, two validation tools are available (also called inside `run-commands.sh`):

**Check structural correctness** — verifies that removing gaps from the alignment exactly recovers the original sequences:
```bash
./build/check_alignment --raw data/sequences.fa --alignment build/alignment.fa
# Add -v to print individual failures
```

**Compare alignment quality** — scores your alignment against the reference and reports percentage score loss:
```bash
./build/compare_alignment --reference data/hmm_fullDP_ref.fa --estimate build/alignment.fa
# Add -v to see pair-wise score comparisons
```

For the SARS-CoV-2 dataset, substitute `sars_20000.fa` and `sars_20000_ref.fa` accordingly.

---

## Performance Summary

| Mode | Runtime (Dataset 1, 5000 pairs) | Score Loss |
|------|---------------------------------|------------|
| CPU Full DP | ~3,600,000 ms (~1 hr) | 0.000% |
| CPU GACT | ~90,749 ms | 0.056% |
| GPU Single Thread | ~4,035,560 ms | 0.056% |
| GPU Level 1 | ~43,716 ms | 0.056% |
| GPU Level 2 | ~10,844 ms | 0.056% |
| GPU Level 3 (final) | ~4,200 ms | 0.056% |

All experiments were run on the UCSD DSMLP cluster using an NVIDIA A30 GPU (Ampere, 56 SMs, 24 GB HBM2) inside the `yatisht/ece213-wi26:latest` container.

---

## Notes

- The `.h` and `.cuh` header files do **not** need to be modified for any mode, including those in `submission-files/`.
- The `submission-files/` directory contains all implementation variants as standalone files. Always **copy** them into `src/` rather than editing them in place.
- SARS-CoV-2 sequences contain IUPAC ambiguity characters (e.g., `N`, `W`, `R`) which cause a higher score loss (~3.6%) relative to the full-DP reference. This is expected behaviour, not a bug.
