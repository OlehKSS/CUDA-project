#include "bitonic.cuh"

__global__ void BitonicMergeSort(float * d_output, float * d_input, int subarray_size)
{
	extern __shared__ float shared_data[];
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	double portions = log2(double(subarray_size)) - 1;

	shared_data[index] = d_input[index];
	__syncthreads();

	for (short portion = 0; portion <= portions; portion++)
	{
		short offset = 1 << portion;
		short threads_in_box = offset << 1;
		// calculated at the beginning of each portion
		//int boxI = index % (threads_in_box + (blockDim.x * blockIdx.x));
		int boxI = threadIdx.x / threads_in_box;
		for (short subportion = portion; subportion >= 0; subportion--)
		{
			offset = 1 << subportion;
			threads_in_box = offset << 1;
			int arrow_bottom = index % threads_in_box;

			if (((boxI + 1) % 2) == 1) {
				// top down
				if (arrow_bottom < offset) {
					float temp = shared_data[index];
					if (shared_data[index + offset] < temp) {
						shared_data[index] = shared_data[index + offset];
						shared_data[index + offset] = temp;
					}
				}
			}
			else {
				// bottom up
				if (arrow_bottom >= offset) {
					float temp = shared_data[index];
					if (shared_data[index - offset] < temp) {
						shared_data[index] = shared_data[index - offset];
						shared_data[index - offset] = temp;
					}
				}
			}
			__syncthreads();
		}
	}

	d_output[index] = shared_data[index];
}