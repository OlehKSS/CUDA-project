#include "merge_sort.cuh"

void mergeSortAsc(int* arr, int length, int *out)
{
	if (length < 2) {
		out[0] = arr[0];
		return;
	}
	//splitting of the arrays
	int halfSize = length / 2;

	int length_left = halfSize;
	int length_right = length - halfSize;

	int *leftPart = new int[length_left];
	int *rightPart = new int[length_right];

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

	int* out_left = new int[length_left];
	int* out_right = new int[length_right];

	mergeSortAsc(leftPart, length_left, out_left);
	mergeSortAsc(rightPart, length_right, out_right);

	int* out_temp = new int[length];
	
	mergeArraysAsc(out_left, out_right, length_left, length_right, out_temp);

	for (int i = 0; i < length; i++)
	{
		out[i] = out_temp[i];
	}
}

void mergeArraysAsc(int* arr_left, int* arr_right, int length_left, int length_right, int* out)
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