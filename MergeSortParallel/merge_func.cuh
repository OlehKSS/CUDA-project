#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void MergeRank(float * d_input, float * d_output);

void orderBitonicArray(float* d_in, int size, int part_size, float* d_out, bool log=false);
__global__ void mergingKernel(float* in_array, int part_size, float* out_array);
__device__ void mergeArraysAsc(float* arr_left, float* arr_right, int length_left, int length_right, float* out, int out_shift);