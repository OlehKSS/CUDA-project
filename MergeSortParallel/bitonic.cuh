#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <math.h>

__global__ void BitonicMergeSort(float * d_output, float * d_input, int subarray_size);