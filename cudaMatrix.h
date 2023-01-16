
#include <memory>

struct Shape{
	size_t x, y;

	Shape(size_t x = 1, size_t y = 1);
};


class cudaMatrix {
private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();

public:
	Shape shape;

	std::shared_ptr<float> data_device;
	std::shared_ptr<float> data_host;

	cudaMatrix(size_t x_dim = 1, size_t y_dim = 1);
	cudaMatrix(Shape shape);

	void allocateMemory();
	void allocateMemoryIfNotAllocated(Shape shape);

	void copyHostToDevice();
	void copyDeviceToHost();

	float& operator[](const int index);
	const float& operator[](const int index) const;
};