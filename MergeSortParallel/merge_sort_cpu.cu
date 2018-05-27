#include "merge_sort_cpu.cuh"

void mergeSortAscCpu(float* arr, int length, float* out)
{
	if (length < 2) {
		out[0] = arr[0];
		return;
	}
	//splitting of the arrays
	int halfSize = length / 2;

	int length_left = halfSize;
	int length_right = length - halfSize;

	float* leftPart = new float[length_left];
	float* rightPart = new float[length_right];

	for (int i = 0; i < length; i++)
	{
		if (i < halfSize)
		{
			//copying of the left part
			leftPart[i] = arr[i];
		}
		else {
			//copying of the right part
			rightPart[i - halfSize] = arr[i];
		}

	}

	float* out_left = new float[length_left];
	float* out_right = new float[length_right];

	mergeSortAscCpu(leftPart, length_left, out_left);
	mergeSortAscCpu(rightPart, length_right, out_right);

	float* out_temp = new float[length];
	
	mergeArraysAscCpu(out_left, out_right, length_left, length_right, out_temp);

	for (int i = 0; i < length; i++)
	{
		out[i] = out_temp[i];
	}
}

void mergeArraysAscCpu(float* arr_left, float* arr_right, int length_left, int length_right, float* out)
{
	int totalLength = length_left + length_right;

	//running indices
	int i = 0;
	int j = 0;
	int index = 0;

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