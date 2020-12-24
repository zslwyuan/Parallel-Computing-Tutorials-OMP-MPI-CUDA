# Parallel-Computing-Tutorials-OMP-MPI-CUDA

if the code is helpful, a star could be an encouragement for me ^\_^

These are the implementations of the previous assignments from the course COMP 5212 Parallel Computing. Various solutions have been tried and current source codes are based on those get the highest performance on the server. These source codes cover the range from OpenMP, MPI to CUDA. Thanks to Prof. LUO's detailed lectures and TAs' patient guide.

The assignments are required to solve the shortest path problem and Bellman-ford algorithm has been involved, considering that there could be negative circles in the graph. Actually SPFA can get better performance, with multi-thread programming (pthread). However, for CUDA, OMP and MPI, the gap between bellman-ford and SPFA is not that wide for general graphs.

## The major optimizations include:

1) reduce the data transferring, especially those redundant, between hosts and devices
2) heuristic optimizations are involved to terminate the search in advance
3) reduce the trigerring of the "critical" parts in OpenMP
4) make full use of the reduction operation in MPI

