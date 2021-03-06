#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include "thrust\device_vector.h"
#include "thrust\sort.h"

#include <algorithm>
#include <ctime>
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include "bitonic.cuh"
#include "merge_func.cuh"
#include "merge_sort_cpu.cuh"

int main(int argc, char **argv)
{
	int n_el = 1048575;
	int subarray_size = 1024;

	if (argc >= 3)
	{
		std::istringstream arg1(argv[1]);
		std::istringstream arg2(argv[2]); 
		arg1 >> n_el;
		arg2 >> subarray_size;
	}

	// Compare our results with thrust

    int ARRAY_SIZE = pow(2, ceil(log(n_el)/log(2)));

    int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);


    // generate the input array on the host
    float *h_input = new float[ARRAY_SIZE];
    float *h_output = new float[ARRAY_SIZE];
	//this array in not limited to the power of two sizes
	float *h_output_cpu = new float[n_el];

    for(int i = 0; i < n_el; i++) {
        // generate random float in [0, 999]
        //h_input[i] = (float)rand()/(float)RAND_MAX;
		h_input[i] = rand()%10+1;
    }
	for(int i = n_el; i < ARRAY_SIZE; i++) {
        // generate random float in [0, 999]
        //h_input[i] = (float)rand()/(float)RAND_MAX;
		h_input[i] = 0;
    }

	auto start_cpu = std::chrono::high_resolution_clock::now();
	mergeSortAscCpu(h_input, n_el, h_output_cpu);


	auto end_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time_span_cpu_ms = end_cpu - start_cpu;

	std::cout << "--------------------------CPU---------------------------" << std::endl;

	//std::copy(h_output_cpu, h_output_cpu + ARRAY_SIZE, std::ostream_iterator<float>(std::cout, " "));
	//std::cout << std::endl;

	std::cout << std::endl << "Merge sort, CPU time elapsed (millisec) "
		<< time_span_cpu_ms.count() << std::endl
		<< "--------------------------CPU---------------------------" << std::endl;

	//thrust sorting
	cudaEvent_t start_thr, stop_thr;
	cudaEventCreate(&start_thr);
	cudaEventCreate(&stop_thr);

	cudaEventRecord(start_thr);

	thrust::device_vector<float> d_input_thrust(ARRAY_SIZE);
	float* host_temp_space = new float[ARRAY_SIZE];
	thrust::copy(h_input, h_input + ARRAY_SIZE, d_input_thrust.begin());

	thrust::sort(d_input_thrust.begin(), d_input_thrust.end());

	thrust::copy(d_input_thrust.begin(), d_input_thrust.end(), host_temp_space);

	cudaEventRecord(stop_thr);
	cudaEventSynchronize(stop_thr);
	float milliseconds_thr = 0;
	cudaEventElapsedTime(&milliseconds_thr, start_thr, stop_thr);

	delete[] host_temp_space;
	std::cout << "-----------------------THRUST---------------------------" << std::endl;
	//thrust::copy(d_input_thrust.begin(), d_input_thrust.end(), std::ostream_iterator<float>(std::cout, " "));
	std::cout << "Thrust sorting, GPU time elapsed (millisec) " << milliseconds_thr << std::endl;
	std::cout << "-----------------------THRUST---------------------------" << std::endl;

	cudaProfilerStart();

    // declare GPU memory pointers
    float * d_input, * d_output, *d_output_part;

    // allocate GPU memory
    cudaMalloc((void **) &d_input, ARRAY_BYTES);
    cudaMalloc((void **) &d_output, ARRAY_BYTES);
	cudaMalloc((void **)&d_output_part, ARRAY_BYTES);
   

    // launch the kernel
	int threads_per_block = ARRAY_SIZE;
	int num_blocks = int((ARRAY_SIZE-1)/1024) + 1;
	if(ARRAY_SIZE > 1024)
		threads_per_block = 1024;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	// transfer the input array to the GPU
    cudaMemcpy(d_input, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice);

	int num_of_blocks = static_cast<int>(ARRAY_SIZE/ subarray_size);

	std::cout << "-------------------GPU MERGE SORT-----------------------" << std::endl;
	std::cout << "Number of threads\t" << subarray_size << std::endl;
	std::cout << "Number of blocks\t" << num_of_blocks << std::endl;

    BitonicMergeSort<<<num_of_blocks, subarray_size, subarray_size * sizeof(float)>>>(d_output_part, d_input, subarray_size);
	orderBitonicArray(d_output_part, ARRAY_SIZE, subarray_size, d_output, false);

	// copy back the sum from GPU
    cudaMemcpy(h_output, d_output, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Merge sort, GPU time elapsed (millisec) " << milliseconds << std::endl;
	//std::copy(h_output, h_output + ARRAY_SIZE, std::ostream_iterator<float>(std::cout, " "));
	//std::cout << std::endl;
	std::cout << "-------------------GPU MERGE SORT-----------------------" << std::endl;

	//getchar();
	cudaProfilerStop();

    // free GPU memory allocation
	delete[] h_input;
	delete[] h_output;
	delete[] h_output_cpu;
    cudaFree(d_input);
    cudaFree(d_output);
	cudaFree(d_output_part);

    return 0;
}