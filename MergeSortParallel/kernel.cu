#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <ctime>
#include <iostream>




__global__ void BitonicMergeSort(float * d_output, float * d_input, int subarray_size)
{
    extern __shared__ float shared_data[];
    int index  = blockDim.x * blockIdx.x + threadIdx.x;
	double portions = log2(double(subarray_size))-1;

    shared_data[index] = d_input[index];
    __syncthreads();

    for (short portion = 0; portion <= portions; portion++)
    {
        short offset = 1<<portion;
        short threads_in_box = offset<<1;
        // calculated at the beginning of each portion
        //int boxI = index % (threads_in_box + (blockDim.x * blockIdx.x));
		int boxI = threadIdx.x/threads_in_box;
        for (short subportion = portion; subportion >= 0; subportion--)
        {
            offset = 1<<subportion;
            threads_in_box = offset<<1;
            int arrow_bottom = index % threads_in_box;

            if (((boxI + 1) % 2) == 1) {
                // top down
                if (arrow_bottom < offset) {
                    float temp = shared_data[index];
                    if (shared_data[index+offset] < temp) {
                        shared_data[index] = shared_data[index+offset];
                        shared_data[index+offset] = temp;
                    }
                }
            } else {
                // bottom up
                if (arrow_bottom >= offset) {
                    float temp = shared_data[index];
                    if (shared_data[index-offset] < temp) {
                        shared_data[index] = shared_data[index-offset];
                        shared_data[index-offset] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    d_output[index] = shared_data[index];

}


int main(int argc, char **argv)
{
	int n_el = 8192;

    int ARRAY_SIZE = pow(2, ceil(log(n_el)/log(2)));;
    int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);


    // generate the input array on the host
    float *h_input = new float[ARRAY_SIZE];
    float *h_output = new float[ARRAY_SIZE];
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

    // declare GPU memory pointers
    float * d_input, * d_output;

    // allocate GPU memory
    cudaMalloc((void **) &d_input, ARRAY_BYTES);
    cudaMalloc((void **) &d_output, ARRAY_BYTES);

    

    // launch the kernel
	int threads_per_block = ARRAY_SIZE;
	int num_blocks = int((ARRAY_SIZE-1)/1024) + 1;
	if(ARRAY_SIZE > 1024)
		threads_per_block = 1024;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// transfer the input array to the GPU
    cudaMemcpy(d_input, h_input, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	int subarray_size = 256;

    BitonicMergeSort<<<32, subarray_size, ARRAY_SIZE * sizeof(float)>>>(d_output, d_input, subarray_size);

	cudaEventRecord(stop);

	// copy back the sum from GPU
    cudaMemcpy(h_output, d_output, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

	//for(int i = 0; i < ARRAY_SIZE; i++){
	//	printf("%f \n",h_input[i]);
	//}
	//printf("\n\n");
	float *final_array = h_output+(ARRAY_SIZE-n_el);

	for(int i = 0; i < n_el; i++){
		printf("%.0f ",final_array[i]);
	}

	std::cout << "Merge sort, GPU time elapsed (millisec) " << milliseconds << std::endl;
	getchar();

    // free GPU memory allocation
	delete[] h_input;
	delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}