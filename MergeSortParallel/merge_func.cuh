#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void MergeRank(float * d_input, float * d_output);

void orderBitonicArray(int* d_in, int size, int part_size, int* d_out, bool log=false);
__global__ void mergingKernel(int* in_array, int part_size, int* out_array);
__device__ void mergeArraysAsc(int* arr_left, int* arr_right, int length_left, int length_right, int* out, int out_shift);