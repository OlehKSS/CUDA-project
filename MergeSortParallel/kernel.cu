#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <chrono>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bitonic.cuh"
#include "merge_func.cuh"
#include "merge_sort_cpu.cuh"

int main(int argc, char **argv)
{
	int n_el = 8192;
	int ARRAY_SIZE = n_el;

    //int ARRAY_SIZE = pow(2, ceil(log(n_el)/log(2)));

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
	/*for(int i = n_el; i < ARRAY_SIZE; i++) {
        // generate random float in [0, 999]
        //h_input[i] = (float)rand()/(float)RAND_MAX;
		h_input[i] = 0;
    }*/

	auto start_cpu = std::chrono::high_resolution_clock::now();
	mergeSortAscCpu(h_input, n_el, h_output_cpu);


	auto end_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time_span_cpu_ms = end_cpu - start_cpu;

	std::cout << "-----------------------CPU---------------------------" << std::endl;

	/*for (int i = 0; i < n_el; i++) {
		printf("%.0f ", h_output_cpu[i]);
	}*/

	std::cout << std::endl << "Merge sort, CPU time elapsed (millisec) "
		<< time_span_cpu_ms.count() << std::endl
		<< "-----------------------CPU---------------------------" << std::endl;

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

	int subarray_size = 1024;
	int num_of_blocks = static_cast<int>(ARRAY_SIZE/ subarray_size);

    BitonicMergeSort<<<num_of_blocks, subarray_size, ARRAY_SIZE * sizeof(float)>>>(d_output_part, d_input, subarray_size);
	orderBitonicArray(d_output_part, ARRAY_SIZE, subarray_size, d_output);

	// copy back the sum from GPU
    cudaMemcpy(h_output, d_output, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

	//for(int i = 0; i < ARRAY_SIZE; i++){
	//	printf("%f \n",h_input[i]);
	//}
	//printf("\n\n");
	//float *final_array = h_output+(ARRAY_SIZE-n_el);

	std::cout << "Merge sort, GPU time elapsed (millisec) " << milliseconds << std::endl;
	for(int i = 0; i < n_el; i++){
		printf("%.0f ", h_output[i]);
	}

	getchar();

    // free GPU memory allocation
	delete[] h_input;
	delete[] h_output;
	delete[] h_output_cpu;
    cudaFree(d_input);
    cudaFree(d_output);
	cudaFree(d_output_part);

    return 0;
}