#ifndef MATRIX_H
#define MATRIX_H

#include "cudaMatrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

typedef struct
{
    double * m;
    unsigned columns;
    unsigned rows;
}  matrix_t;

matrix_t * alloc_matrix(unsigned rows, unsigned columns);

void destroy_matrix(matrix_t *m);

void print_matrix(cudaMatrix *m, bool is_short);

void hadamard_product(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res);

void matrix_sum(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res);

void matrix_minus(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res);

void matrix_dot(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res);

void matrix_function(cudaMatrix *m1, double (*f)(double), cudaMatrix *res);

void matrix_transpose(cudaMatrix *m1, cudaMatrix *res);

void matrix_scalar(cudaMatrix *m1, double s, cudaMatrix *res);

void matrix_memcpy(cudaMatrix *dest, const cudaMatrix *src);

void matrix_sum_Kernel(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res);

__global__ void matrix_minus_Kernel(double *m1, double *m2, double *res, int rows, int col);

void matrix_scalar_Kernel(cudaMatrix *m1, double s, cudaMatrix *res);

__global__ void matrix_function_Kernel(double *m1, double (*f)(double), double *res, int rows, int col);

// __global__ void matrix_dot_Kernel(cudaMatrix *m1, cudaMatrix *m2, cudaMatrix *res);

// __global__ void matrix_transpose_Kernel(cudaMatrix *m1, cudaMatrix *res);


#endif