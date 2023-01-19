#ifndef ANN_H
#define ANN_H

#include "cudaMatrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "matrix.h"

typedef struct 
{
    unsigned minibatch_size;
    unsigned number_of_neurons;
    
    cudaMatrix* weights;
    cudaMatrix* biases;

    cudaMatrix* z;
    cudaMatrix* activations;
    
    cudaMatrix* delta;
} layer_t;

typedef struct 
{
    void (*f)(double*, double*, unsigned, unsigned);
    void (*fd)(double*, double*, unsigned, unsigned);
    double alpha;
    unsigned minibatch_size;
    unsigned input_size;
    unsigned number_of_layers;
    layer_t** layers;
} ann_t;

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer);

layer_t * create_layer(unsigned l, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size);

void set_input(ann_t *nn, cudaMatrix* input);

void print_nn(ann_t *nn);

void forward(ann_t *nn, double (*activation_function)(double));

void backward(ann_t *nn, cudaMatrix* cy, double (*derivative_actfunct)(double));

#endif