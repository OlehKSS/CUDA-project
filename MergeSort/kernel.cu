
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <iostream>
#include <stdio.h>

#include "merge_sort.cuh"


void init_array(int* in, int size, int max_level);
void print_vector(int* in, int size);

int main()
{
	int size = 100;
	int max_val = 100;

	int* test = new int[size];
	int* out = new int[size];

	init_array(test, size, max_val);
	print_vector(test, size);

	auto start_cpu = std::chrono::high_resolution_clock::now();

	mergeSortAsc(test, size, out);

	auto end_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span_cpu = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu - start_cpu);
	
	print_vector(out, size);
	std::cout << "Merge sort, CPU time elapsed (millisec) " << time_span_cpu.count() << std::endl;

	system("pause");
	return 0;
}

void init_array(int* in, int size, int max_level)
{
	for (int i = 0; i < size; i++)
	{
		in[i] = floor(max_level*((double)rand() / (RAND_MAX)));
	}
}

void print_vector(int* in, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cout << in[i] << " ";
	}

	std::cout << std::endl;
}
