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

// __device__ void matrix_sum_Kernel(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
// {
//     assert ( (m1->columns == m2->columns)  &&
//              (m1->columns == res->columns) &&
//              (m1->rows == m2->rows)        &&
//              (m1->rows == res->rows));

//     unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

//     if (idx < m1->rows * m1->columns)
//     { 
//         res->data_device.get()[idx] = m1->data_device.get()[idx] + m2->data_device.get()[idx];
//     }
// }

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

// __global__ void matrix_minus_Kernel(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res)
// {
//     assert ( (m1->columns == m2->columns)  &&
//              (m1->columns == res->columns) &&
//              (m1->rows == m2->rows)        &&
//              (m1->rows == res->rows));

//     unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

//     if (idx < m1->rows * m1->columns)
//     { 
//         res->data_device.get()[idx] = m1->data_device.get()[idx] - m2->data_device.get()[idx];
//     }
// }

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

void matrix_function(cudaMatrix *m1, double (*f)(double), cudaMatrix *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        (*res)[idx] = f((*m1)[idx]);
    }
}

// __global__ void matrix_function_Kernel(cudaMatrix *m1, double (*f)(double), cudaMatrix *res)
// {
//     assert ( (m1->columns == res->columns) &&             
//             (m1->rows == res->rows));

//     unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

//     if (idx < m1->rows * m1->columns)
//     { 
//         res->data_device.get()[idx] = f(m1->data_device.get()[idx]);
//     }
// }

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

void matrix_scalar(cudaMatrix *m1, double s, cudaMatrix *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        (*res)[idx] = (*m1)[idx] * s;
    }
}

// __global__ void matrix_scalar_Kernel(cudaMatrix *m1, double s, cudaMatrix *res)
// {
//     assert ( (m1->rows == res->rows) &&             
//              (m1->columns == res->columns));

//     unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

//     if (idx < m1->rows * m1->columns)
//     { 
//         res->data_device.get()[idx] = m1->data_device.get()[idx] * s;
//     }
// }

void matrix_memcpy(cudaMatrix *dest, const cudaMatrix *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->data_host, src->data_host, src->columns * src->rows * sizeof(double));     
}