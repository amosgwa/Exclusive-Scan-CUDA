# Exclusive-Scan-CUDA
Implementation of Exclusive Scan and Reduce using CUDA and serial implementation.
The program generate N sized array with random numbers [1,1000]. Then, it computes the exculsive scan array using both Serial and Parallel techniques. For pralllelization, I used CUDA, and the thread size is 1024.

# Compile using
nvcc main.cu -o program

# Usage
program N

# To improve
Currently it only passes for N < thread size^2. Increasing the threadsize will improve the limitation. 
