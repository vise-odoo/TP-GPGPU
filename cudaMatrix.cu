#include "cudaMatrix.h"

cudaMatrix* initCudaMatrix(unsigned rows, unsigned columns) {
	cudaMatrix* m = (cudaMatrix*) malloc( sizeof(cudaMatrix) );
	*m = cudaMatrix(rows, columns);
	m->allocateMemory();
	return m;
}

cudaMatrix::cudaMatrix(unsigned rows, unsigned columns) :
	rows(rows), columns(columns), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{ }


void cudaMatrix::allocateCudaMemory() {
	if (!device_allocated) {
		cudaMalloc(&data_device, rows * columns * sizeof(double));
		device_allocated = true;
	}
}

void cudaMatrix::allocateHostMemory() {
	if (!host_allocated) {
		data_host = (double *) calloc(columns * rows, sizeof(double));
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
		cudaMemcpy(data_device, data_host, rows * columns * sizeof(double), cudaMemcpyHostToDevice);
	}
}

void cudaMatrix::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_host, data_device, rows * columns * sizeof(double), cudaMemcpyDeviceToHost);
	}
}

double& cudaMatrix::operator[](const int index) {
	return data_host[index];
}

const double& cudaMatrix::operator[](const int index) const {
	return data_host[index];
}

void cudaMatrix::destroyCudaMatrix(){
	cudaFree(data_device);
	free(data_host);
}
