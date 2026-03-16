#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

mkdir -p build
cd build
cmake ..
make -j4
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/tbb_cmake_build/tbb_cmake_build_subdir_release



## Basic run to align the first 10 pairs (20 sequences) in the sequences.fa in batches of 2 reads
## HINT: may need to change values for the assignment tasks. You can create a sequence of commands

#Uncomment for Sequences dataset
time ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --batchSize 625 --output alignment.fa
#Uncomment for Sars Cov2 Dataset
# time ./aligner --sequence ../data/sars_20000.fa --maxPairs 1000 --batchSize 125 --output alignment.fa



## Debugging with Compute Sanitizer
## Run this command to detect illegal memory accesses (out-of-bounds reads/writes) and race conditions.
# compute-sanitizer ./aligner --sequence ../data/sequences.fa --maxPairs 5000 --output alignment.fa



## For debugging and evaluate the alignment accuracy

#Uncomment for Sequences dataset
./check_alignment --raw ../data/sequences.fa --alignment alignment.fa                  # add -v to see failed result instead of a summary
./compare_alignment --reference ../data/hmm_fullDP_ref.fa --estimate alignment.fa # add -v to see pair-wise comparisons
#Uncomment for Sars Cov2 Dataset
# ./check_alignment --raw ../data/sars_20000.fa --alignment alignment.fa                  # add -v to see failed result instead of a summary
# ./compare_alignment --reference ../data/sars_20000_ref.fa --estimate alignment.fa # add -v to see pair-wise comparisons
