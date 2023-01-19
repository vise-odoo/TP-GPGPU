#include "cudaMatrix.h"

cudaMatrix::cudaMatrix(unsigned rows, unsigned columns) :
	rows(rows), columns(columns), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{ }


void cudaMatrix::allocateCudaMemory() {
	if (!device_allocated) {
		double* device_memory = nullptr;
		cudaMalloc(&device_memory, rows * columns * sizeof(double));
		data_device = std::shared_ptr<double>(device_memory,
											 [&](double* ptr){ cudaFree(ptr); });
		device_allocated = true;
	}
}

void cudaMatrix::allocateHostMemory() {
	if (!host_allocated) {
		data_host = std::shared_ptr<double>(new double[rows * columns],
										   [&](double* ptr){ delete[] ptr; });
		host_allocated = true;
	}
}

void cudaMatrix::allocateMemory() {
	allocateCudaMemory();
	allocateHostMemory();
}

void cudaMatrix::allocateMemoryIfNotAllocated(unsigned rows, unsigned columns) {
	if (!device_allocated && !host_allocated) {
		this->rows = rows;
        this->columns = columns;
		allocateMemory();
	}
}

void cudaMatrix::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_device.get(), data_host.get(), rows * columns * sizeof(double), cudaMemcpyHostToDevice);
	}
}

void cudaMatrix::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_host.get(), data_device.get(), rows * columns * sizeof(double), cudaMemcpyDeviceToHost);
	}
}

double& cudaMatrix::operator[](const int index) {
	return data_host.get()[index];
}

const double& cudaMatrix::operator[](const int index) const {
	return data_host.get()[index];
}

void cudaMatrix::destroyCudaMatrix(){
	cudaFree(data_device.get());
	free(data_host.get());
}
