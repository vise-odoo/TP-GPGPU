#ifndef CUDAMATRIX_H
#define CUDAMATRIX_H

#include <memory>
class cudaMatrix {
private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();

public:
	unsigned rows;
    unsigned columns;

	double* data_device;
	double* data_host;

	cudaMatrix(unsigned rows, unsigned columns);

	void allocateMemory();
	void allocateMemoryIfNotAllocated(unsigned rows, unsigned columns);

	void copyHostToDevice();
	void copyDeviceToHost();
	
	void destroyCudaMatrix();

	double& operator[](const int index);
	const double& operator[](const int index) const;
};

cudaMatrix* initCudaMatrix(unsigned rows, unsigned columns);

#endif