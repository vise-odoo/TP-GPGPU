#include "matrix.h"
#include "cudaMatrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// TODO : matrix dot , matrix hadamard, matrix transpose
matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    free(m->m);
    free(m);
}

void print_matrix(cudaMatrix *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", (*m)[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            (*res)[idx] = (*m1)[idx] * (*m2)[idx];
    }
}

__global__ void hadamard_product_Device(double *m1, double *m2, double *res, int rows, int col)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < rows * col)
    { 
        res[idx] = m1[idx] * m2[idx];
    }
}

void hadamard_product_Kernel(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    m1->copyHostToDevice();
    m2->copyHostToDevice();
    hadamard_product_Device<<<8, 1024>>>(m1->data_device, m2->data_device, res->data_device, m1->rows, m1->columns);
    res->copyDeviceToHost();
}


void matrix_sum(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        (*res)[idx] = (*m1)[idx] + (*m2)[idx];
    }
}

__global__ void matrix_sum_Device(double *m1, double *m2, double *res, int rows, int col)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < rows * col)
    { 
        res[idx] = m1[idx] + m2[idx];
    }
}

void matrix_sum_Kernel(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    m1->copyHostToDevice();
    m2->copyHostToDevice();

    matrix_sum_Device<<<8, 1024>>>(m1->data_device, m2->data_device, res->data_device, m1->rows, m1->columns);

    res->copyDeviceToHost();
}

void matrix_minus(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        (*res)[idx] = (*m1)[idx] - (*m2)[idx];
    }
}

__global__ void matrix_minus_Device(double *m1, double *m2, double *res, int rows, int col)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < rows * col)
    { 
        res[idx] = m1[idx] - m2[idx];
    }
}

void matrix_minus_Kernel(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    m1->copyHostToDevice();
    m2->copyHostToDevice();

    matrix_minus_Device<<<8, 1024>>>(m1->data_device, m2->data_device, res->data_device, m1->rows, m1->columns);

    res->copyDeviceToHost();
}

void matrix_dot(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += (*m1)[ii + row * m1->columns] * (*m2)[col + ii * m2->columns];
            }

            (*res)[idx] = var;
        }
    }
}

__global__ void matrix_dot_Device(double *m1,double *m2, double *res, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += m1[row * n + i] * m2[i * k + col];
        }
        res[row * k + col] = sum;
    }
} 

void matrix_dot_Kernel(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
{
    dim3 dimGrid(8,8,1);
    dim3 dimBlock(32,32,1);

    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    m1->copyHostToDevice();
    m2->copyHostToDevice();
    matrix_dot_Device<<<dimGrid, dimBlock>>>(m1->data_device, m2->data_device, res->data_device, m1->rows,m1->columns, m2->columns);
    res->copyDeviceToHost();
}

void matrix_function(cudaMatrix *m1, double (*f)(double), cudaMatrix *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        (*res)[idx] = f((*m1)[idx]);
    }
}

__device__ double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

__device__ double dsigmoid(double x)
{
    return sigmoid(x)*(1-sigmoid(x));
}

__global__ void matrix_function_Device(double *m1, int fn, double *res, int rows, int col)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < rows * col)
    {
        switch (fn) {
            case 1:
                res[idx] = sigmoid(m1[idx]);
                break;
            case 2:
                res[idx] = dsigmoid(m1[idx]);
                break;
            default:
                res[idx] = 0;
        }
    }
}

void matrix_function_Kernel(cudaMatrix *m1, int fn, cudaMatrix *res)
{
    // double (*f_device)(double);

    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));


    // cudaMalloc((void**)&f_device, sizeof(f));
    // cudaMemcpy(f_device, f, sizeof(f), cudaMemcpyHostToDevice);
    m1->copyHostToDevice(); 
    matrix_function_Device<<<8, 1024>>>(m1->data_device, fn, res->data_device, m1->rows, m1->columns);
    res->copyDeviceToHost();

    // cudaFree(f_device);
}

void matrix_transpose(cudaMatrix *m1, cudaMatrix *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            (*res)[row + col * m1->rows] = (*m1)[col + row * m1->columns];
        }
    }
}

__global__ void matrix_transpose_Device(double* m1, double* res, int rows, int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int index = idy * cols + idx;
        unsigned int idx_transpose = idx * rows + idy;
        m1[idx_transpose] = res[index];
    }
}

__global__ void matrix_transpose_shared_Device(double* m1, double* res, int rows, int cols)
{
	__shared__ float shared[32][33]; // 33 car il peut y avoir des erreurs de sortie de tableaux parfois
	
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if((xIndex < rows) && (yIndex < cols))
	{
        // Ajout des données dans la mémoire partagée, de façon non transposée
		unsigned int idx = yIndex * rows + xIndex; 
		shared[threadIdx.y][threadIdx.x] = m1[idx];
	}

	__syncthreads();
    // On recalcule les indices
	xIndex = blockIdx.y * blockDim.x + threadIdx.x;
	yIndex = blockIdx.x * blockDim.y + threadIdx.y;

	if((xIndex < cols) && (yIndex < rows))
	{
		unsigned int idx_transpose = yIndex * cols + xIndex;
        // Copie des données en transposant le résultat.
		res[idx_transpose] = shared[threadIdx.x][threadIdx.y];
	}
}

void matrix_transpose_Kernel(cudaMatrix *m1, cudaMatrix *res)
{
    dim3 dimGrid(8,8,1);
    dim3 dimBlock(32,32,1);

    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    m1->copyHostToDevice();
    matrix_transpose_Device<<<dimGrid, dimBlock>>>(m1->data_device, res->data_device, m1->rows, m1->columns);
    res->copyDeviceToHost();
}


void matrix_scalar(cudaMatrix *m1, double s, cudaMatrix *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        (*res)[idx] = (*m1)[idx] * s;
    }
}

__global__ void matrix_scalar_Device(double *m1, double s, double *res, int rows, int col)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < rows * col)
    { 
        res[idx] = m1[idx] * s;
    }
}

void matrix_scalar_Kernel(cudaMatrix *m1, double s, cudaMatrix *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    m1->copyHostToDevice();
    matrix_scalar_Device<<<8, 1024>>>(m1->data_device, s, res->data_device, m1->rows, m1->columns);
    res->copyDeviceToHost();
}

void matrix_memcpy(cudaMatrix *dest, const cudaMatrix *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->data_host, src->data_host, src->columns * src->rows * sizeof(double));     
}