#ifndef MATRIX_H
#define MATRIX_H
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

__device__ matrix_t * d_alloc_matrix(unsigned rows, unsigned columns);

void destroy_matrix(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res);

void matrix_transpose(matrix_t *m1, matrix_t *res);

void matrix_scalar(matrix_t *m1, double s, matrix_t *res);

void matrix_memcpy(matrix_t *dest, const matrix_t *src);

__device__ void matrix_sum_Kernel(matrix_t *m1, matrix_t *m2, matrix_t *res);

__device__ void matrix_minus_Kernel(matrix_t *m1, matrix_t *m2, matrix_t *res);

__device__ void matrix_scalar_Kernel(matrix_t *m1, double s, matrix_t *res);

__device__ void matrix_function_Kernel(matrix_t *m1, double (*f)(double), matrix_t *res);

__device__ void matrix_dot_Kernel(matrix_t *m1, matrix_t *m2, matrix_t *res);

__device__ void destroy_d_matrix(matrix_t *m);

#endif