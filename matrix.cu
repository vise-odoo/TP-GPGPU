#include "matrix.h"
#include "cudaMatrix.h"
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

// __global__ void matrix_dot_Kernel(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
// {
//     assert ( (m1->columns == m2->rows)  &&
//              (m1->rows == res->rows)    &&
//              (m2->columns == res->columns));

//     unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
// }

void matrix_function(cudaMatrix *m1, double (*f)(double), cudaMatrix *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        (*res)[idx] = f((*m1)[idx]);
    }
}

__global__ void matrix_function_Device(double *m1, double (*f)(double), double *res, int rows, int col)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < rows * col)
    { 
        res[idx] = f(m1[idx]);
    }
}

void matrix_function_Kernel(cudaMatrix *m1, double (*f)(double), cudaMatrix *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    m1->copyHostToDevice();
    matrix_function_Device<<<8, 1024>>>(m1->data_device, f, res->data_device, m1->rows, m1->columns);
    res->copyDeviceToHost();
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

// __global__ void matrix_transpose_Kernel(cudaMatrix *m1, cudaMatrix *res)
// {
//     assert ( (m1->columns == res->rows) &&             
//              (m1->rows == res->columns));

//     unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

//     __shared__ double smemArray[1024];

//     if (idx < m1->rows * m1->columns)
//     {
//         smemArray[idx] = (*m1)[idx];
//     }

//     __syncthreads();

//     if (idx < m1->rows * m1->columns)
//     {
//         (*m1)[idx] = smemArray[idx];
//     }
// }

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