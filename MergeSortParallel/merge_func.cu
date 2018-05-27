#include "merge_func.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void MergeRank(float * d_input, float * d_output) 
{
	int indexA = blockIdx.x * blockDim.x + threadIdx.x;
	int indexB = indexA + 2048;
	float temp1 = d_input[indexA];
	float temp2 = d_input[indexB];
	int indexAB = 2048;
	while (d_input[indexAB] < temp1) {
		indexAB++;
	}
	int indexBA = 0;
	while (d_input[indexBA] < temp2) {
		indexBA++;
	}
	__syncthreads();
	d_output[indexA + indexAB + 1] = temp1;
	d_output[indexB + indexBA + 1] = temp2;

}

void orderBitonicArray(int* d_in, int size, int part_size, int* d_out, bool log)
{
	/**
	* \brief Order output array of the bitonic sort function
	* \param d_in - a partially sorted array, global memory, gpu
	* \param size - the size of the input array
	* \param part_size - the size of a sorted subarray
	* \param d_out - a pointer to the output array, global memory, gpu, where
	*    function execution result will be stored
	* \param log - show information about performance during each step
	* \return
	* void
	*/

	int iter_number = static_cast<int>(log2(size / part_size));
	int init_num_threads = size / (2 * part_size);
	int init_num_blocks = ((init_num_threads - 1) / 1024) + 1;

	if (log)
	{
		std::cout << "--------------------------start log--------------------------------" << std::endl;
		std::cout << "Number of steps\t" << iter_number << std::endl;
	}

	int* t_d_in = d_in;

	for (int i = 0; i < iter_number; i++)
	{
		if (log)
		{
			std::cout << "-------------------------------------------------------------------" << std::endl;
			std::cout << "Merging step #" << i << std::endl;
			std::cout << "Number of blocks\t" << init_num_threads << std::endl;
			std::cout << "Number of threads\t" << init_num_threads << std::endl;
		}

		mergingKernel << <init_num_blocks, init_num_threads >> >(t_d_in, part_size, d_out);
		part_size *= 2;
		init_num_threads = init_num_threads / 2;
		init_num_blocks = ((init_num_threads - 1) / 1024) + 1;

		cudaFree(t_d_in);
		cudaMalloc((void **)&t_d_in, size * sizeof(int));
		cudaMemcpy(t_d_in, d_out, size * sizeof(int), cudaMemcpyDeviceToDevice);
		
		if (log)
		{
			int *out = new int[size];
			cudaMemcpy(out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

			for (int i = 0; i < size; i++) {
				std::cout << out[i] << "\t";
			}
			std::cout << std::endl;
			std::cout << std::endl;

			delete[] out;

			if (i == iter_number - 1)
			{
				std::cout << "----------------------------end log--------------------------------" << std::endl;
			}
		}
	}
}

__global__ void mergingKernel(int* in_array, int part_size, int* out_array)
{
	/**
	* \brief kernel function for merging of the arrays of the partially sorted array
	* \param in_array - the input array
	* \param part_size - the size of a sorted subarray
	* \param out_array - a pointer to the output array, 
	*    where result of merging will be stored
	* \return
	* void
	*/
		

	int index = blockDim.x * blockIdx.x + threadIdx.x;

	int* arr_left = in_array + 2 * part_size * index;
	int* arr_right = arr_left + part_size;

	int out_shift = 2 * part_size * index;

	mergeArraysAsc(arr_left, arr_right, part_size, part_size, out_array, out_shift);

	__syncthreads();
}

__device__ void mergeArraysAsc(int* arr_left, int* arr_right, int length_left, int length_right, int* out, int out_shift)
{
	/**
	* \brief Helper function for the mergingKernel function, merges subarrays
	* \param arr_left - the first sorted array
	* \param arr_right -  the second sorted array
	* \param length_left - size of the first array
	* \param length_right - size of the second array
	* \param out - a pointer to the output array, where result will be stored
	* \param out_shift - shift, from which to start writing in output array.
	* \return
	* void
	*/

	int totalLength = length_left + length_right;

	//running indices
	int i = 0;
	int j = 0;
	int index = out_shift;

	while (i < length_left && j < length_right)
	{
		if (arr_left[i] <= arr_right[j])
		{
			out[index] = arr_left[i];
			i++;
			index++;
		}
		else {
			out[index] = arr_right[j];
			j++;
			index++;
		}
	}

	//only one of these two loops will run
	while (i < length_left)
	{
		out[index] = arr_left[i];
		index++;
		i++;
	}

	while (j < length_right)
	{
		out[index] = arr_right[j];
		index++;
		j++;
	}
}