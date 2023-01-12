#define _USE_MATH_DEFINES
#include "ann.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

double normalRand(double mu, double sigma);
void init_weight(matrix_t* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

bool generate = false;
double z1 = 0;

double normalRand(double mu, double sigma)
{
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*M_PI;
    
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   u1 = (double) rand() / RAND_MAX;
	   u2 = (double) rand() / RAND_MAX;
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void init_weight(matrix_t* w, unsigned nneurones_prev)
{
    for (int idx = 0; idx < w->columns * w->rows; idx ++)
    {
        w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
    }
}

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    ann_t * nn = (ann_t *)malloc(sizeof(ann_t));

    nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
    }

    return nn;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t * layer = (layer_t*) malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t* input){
    matrix_memcpy(nn->layers[0]->activations, input);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);
    
    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
    
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward(ann_t *nn, double (*activation_function)(double))
{
    //unsigned int *h_weights, *h_activations , *h_z1 , *h_biases, *h_one, *h_z2 ;
    unsigned int *d_weights, *d_activations , *d_z1, *d_biases, *d_one, *d_z2 ;
    dim3 blocksPerGrid = 20;
    dim3 threadsPerBlock = (32,32,1);

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *z1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *z2 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *one = alloc_matrix(1, nn->minibatch_size);

/*
        // Allocation on the CPU side
        h_weights = (unsigned int*)malloc(nn->layers[l]->weights->rows *nn->layers[l]->weights->columns *sizeof(unsigned int));
        h_activations = (unsigned int*)malloc(nn->layers[l-1]->activations->rows *nn->layers[l-1]->activations->columns *sizeof(unsigned int));
        h_z1 = (unsigned int*)malloc(z1->rows *z1->columns *sizeof(unsigned int));
        h_biases = (unsigned int*)malloc(nn->layers[l]->biases->rows *nn->layers[l]->biases->columns *sizeof(unsigned int));
        h_one = (unsigned int*)malloc(one->rows * one->columns *sizeof(unsigned int));
        h_z2 = (unsigned int*)malloc(z2->rows *z2->columns *sizeof(unsigned int));
*/

        // allocate the memory on the GPU
        cudaMalloc((void**)&d_weights, nn->layers[l]->weights->rows *nn->layers[l]->weights->columns *sizeof(unsigned int));
        cudaMalloc((void**)&d_activations, nn->layers[l-1]->activations->rows *nn->layers[l-1]->activations->columns *sizeof(unsigned int));
        cudaMalloc((void**)&d_z1, z1->rows *z1->columns *sizeof(unsigned int));
        cudaMalloc((void**)&d_biases, nn->layers[l]->biases->rows *nn->layers[l]->biases->columns *sizeof(unsigned int));
        cudaMalloc((void**)&d_one, one->rows * one->columns *sizeof(unsigned int));
        cudaMalloc((void**)&d_z2, z2->rows *z2->columns *sizeof(unsigned int));

        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;

        // copy the arrays to the GPU
        cudaMemcpy( d_weights, nn->layers[l]->weights->m, nn->layers[l]->weights->rows *nn->layers[l]->weights->columns *sizeof(unsigned int), cudaMemcpyHostToDevice );
        cudaMemcpy( d_activations, nn->layers[l-1]->activations->m, nn->layers[l-1]->activations->rows *nn->layers[l-1]->activations->columns *sizeof(unsigned int), cudaMemcpyHostToDevice ); 
        cudaMemcpy( d_z1, z1->m, z1->rows *z1->columns *sizeof(unsigned int), cudaMemcpyHostToDevice ); 
        cudaMemcpy( d_biases, nn->layers[l]->biases->m, nn->layers[l]->biases->rows *nn->layers[l]->biases->columns *sizeof(unsigned int), cudaMemcpyHostToDevice );
        cudaMemcpy( d_one, one->m, one->rows *one->columns *sizeof(unsigned int), cudaMemcpyHostToDevice ); 
        cudaMemcpy( d_z2, z2->m, z2->rows *z2->columns *sizeof(unsigned int), cudaMemcpyHostToDevice );

        matrixMultiplicationKernelUnshared<<<blocksPerGrid, threadsPerBlock>>>(d_weights, d_activations, d_z1, nn->layers[l]->weights->rows, nn->layers[l]->weights->columns, nn->layers[l]->activations->columns); // z1 <- w^l x a^(l-1)
        cudaMemcpy( z1->m, d_z1, z1->rows *z1->columns *sizeof(unsigned int), cudaMemcpyDeviceToHost );
     
        matrixMultiplicationKernelUnshared<<<blocksPerGrid, threadsPerBlock>>>(d_biases, d_one, d_z2, nn->layers[l]->biases->rows, nn->layers[l]->biases->columns, one->columns); // z2 <- b^l x 1 
        cudaMemcpy( z2->m, d_z2, z2->rows *z2->columns *sizeof(unsigned int), cudaMemcpyDeviceToHost );
        
        matrix_sum(z1, z2, nn->layers[l]->z); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1      

        matrix_function(nn->layers[l]->z, activation_function, nn->layers[l]->activations); // a^l = f(z^l)
     
        destroy_matrix(z1);
        destroy_matrix(z2);
        destroy_matrix(one);

        cudaFree(d_weights);
        cudaFree(d_activations);
        cudaFree(d_z1);
        cudaFree(d_biases);
        cudaFree(d_one);
        cudaFree(d_z2);
    }

}

void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{
    unsigned L = nn->number_of_layers-1;

    matrix_t *dfzL = alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus(nn->layers[L]->activations, y, nn->layers[L]->delta);  // delta^(L) = (a^L - y)
    matrix_function(nn->layers[L]->z, derivative_actfunct, dfzL); // f'(z^(L))
    hadamard_product(nn->layers[L]->delta, dfzL, nn->layers[L]->delta); // delta^(L) = (a^L - y) o f'(z^(L))

    destroy_matrix(dfzL);

    for (int l = L; l > 1; l--)
    {
        matrix_t *tw, *delta_tmp, *dfz;
        tw = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        delta_tmp = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);
        dfz = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);

        matrix_transpose(nn->layers[l]->weights, tw); // (w^l)T        
        matrix_dot(tw, nn->layers[l]->delta, delta_tmp); // (w^l)T x delta^l
        matrix_function(nn->layers[l-1]->z, derivative_actfunct, dfz); // f'(z^(l-1))
        hadamard_product(delta_tmp, dfz, nn->layers[l-1]->delta); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))

        destroy_matrix(tw);
        destroy_matrix(delta_tmp);
        destroy_matrix(dfz);
    }

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *w1, *ta;
        w1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->layers[l-1]->number_of_neurons);
        ta = alloc_matrix(nn->minibatch_size, nn->layers[l-1]->number_of_neurons);
        
        matrix_transpose(nn->layers[l-1]->activations, ta); // ta <- (a^(l-1))^T
        matrix_dot(nn->layers[l]->delta, ta, w1); // w1 <- delta^l x (a^(l-1))^T
        matrix_scalar(w1, nn->alpha / nn->minibatch_size, w1); // w1 <- alpha /m . delta^l x (a^(l-1))^T
        matrix_minus(nn->layers[l]->weights, w1, nn->layers[l]->weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T

        destroy_matrix(w1);
        destroy_matrix(ta);

        matrix_t *one, *b1;
        b1 = alloc_matrix(nn->layers[l]->number_of_neurons, 1);
        one = alloc_matrix(nn->minibatch_size, 1);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;

        matrix_dot(nn->layers[l]->delta, one, b1); // b1 <- delta^l x 1^T
        matrix_scalar(b1,  nn->alpha / nn->minibatch_size, b1); // b1 <- alpha / m . delta^l x 1^T
        matrix_minus(nn->layers[l]->biases, b1, nn->layers[l]->biases); // b^l = b^l - alpha / m . delta^l x 1^T
        
        destroy_matrix(one);
        destroy_matrix(b1);
    }
}