#define _USE_MATH_DEFINES
#include "ann.h"
#include "cudaMatrix.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

double normalRand(double mu, double sigma);
void init_weight(cudaMatrix* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

bool generate;
double z1;

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

void init_weight(cudaMatrix* w, unsigned nneurones_prev)
{
    for (int idx = 0; idx < w->columns * w->rows; idx ++)
    {
        (*w)[idx] = normalRand(0, 1 / sqrt(nneurones_prev)); // Le programme s'exécute sur la partie host dans cette version non kernelisée
    }
}

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    generate = false; // Ici ces variables ne sont affectées qu'une seule fois, elles sont utilisées dans normalRand
    z1 = 0;
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
    layer->activations = initCudaMatrix(number_of_neurons, minibatch_size);
    layer->z = initCudaMatrix(number_of_neurons, minibatch_size);
    layer->delta = initCudaMatrix(number_of_neurons, minibatch_size);
    layer->weights = initCudaMatrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = initCudaMatrix(number_of_neurons, 1);
    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, cudaMatrix* input){
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
    for (int l = 1; l < nn->number_of_layers; l++)
    {
        cudaMatrix *z1 = initCudaMatrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        cudaMatrix *z2 = initCudaMatrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        cudaMatrix *one = initCudaMatrix(1, nn->minibatch_size);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            (*one)[idx] = 1.0;

        matrix_dot(nn->layers[l]->weights, nn->layers[l-1]->activations, z1); // z1 <- w^l x a^(l-1)
        matrix_dot(nn->layers[l]->biases, one, z2); // z2 <- b^l x 1        

        z1->copyHostToDevice();
        z2->copyHostToDevice();
        nn->layers[l]->z->copyHostToDevice();
        matrix_sum_Kernel<<<32,32>>>(z1, z2, nn->layers[l]->z); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1    
         nn->layers[l]->z->copyDeviceToHost(); 
         
        matrix_function_Kernel<<<32,32>>>(nn->layers[l]->z, activation_function, nn->layers[l]->activations); // a^l = f(z^l)
        nn->layers[l]->activations->copyDeviceToHost(); 

        z1->destroyCudaMatrix();
        z2->destroyCudaMatrix();
        one->destroyCudaMatrix();
    }
}

void backward(ann_t *nn, cudaMatrix *y, double (*derivative_actfunct)(double))
{
    unsigned L = nn->number_of_layers-1;

    cudaMatrix *dfzL = initCudaMatrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus(nn->layers[L]->activations, y, nn->layers[L]->delta);  // delta^(L) = (a^L - y)
    matrix_function(nn->layers[L]->z, derivative_actfunct, dfzL); // f'(z^(L))
    hadamard_product(nn->layers[L]->delta, dfzL, nn->layers[L]->delta); // delta^(L) = (a^L - y) o f'(z^(L))

    dfzL->destroyCudaMatrix();

    for (int l = L; l > 1; l--)
    {
        cudaMatrix *tw, *delta_tmp, *dfz;
        tw = initCudaMatrix(nn->layers[l-1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        delta_tmp = initCudaMatrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);
        dfz = initCudaMatrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);

        matrix_transpose(nn->layers[l]->weights, tw); // (w^l)T        
        matrix_dot(tw, nn->layers[l]->delta, delta_tmp); // (w^l)T x delta^l

        nn->layers[l-1]->z->copyHostToDevice();
        matrix_function_Kernel<<<32,32>>>(nn->layers[l-1]->z, derivative_actfunct, dfz); // f'(z^(l-1))
        dfz->copyDeviceToHost();

        hadamard_product(delta_tmp, dfz, nn->layers[l-1]->delta); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))

        tw->destroyCudaMatrix();
        delta_tmp->destroyCudaMatrix();
        dfz->destroyCudaMatrix();
    }

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        cudaMatrix *w1, *ta;
        w1 = initCudaMatrix(nn->layers[l]->number_of_neurons, nn->layers[l-1]->number_of_neurons);
        ta = initCudaMatrix(nn->minibatch_size, nn->layers[l-1]->number_of_neurons);
        
        matrix_transpose(nn->layers[l-1]->activations, ta); // ta <- (a^(l-1))^T
        matrix_dot(nn->layers[l]->delta, ta, w1); // w1 <- delta^l x (a^(l-1))^T

        w1->copyHostToDevice();
        matrix_scalar_Kernel<<<32,32>>>(w1, nn->alpha / nn->minibatch_size, w1); // w1 <- alpha /m . delta^l x (a^(l-1))^T
         // Pas de copy device to host, puisque la fonction suivante utilise w1 dans le device

        nn->layers[l]->weights->copyHostToDevice();
        matrix_minus_Kernel<<<32,32>>>(nn->layers[l]->weights, w1, nn->layers[l]->weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T
        nn->layers[l]->weights->copyDeviceToHost();

        w1->destroyCudaMatrix();
        ta->destroyCudaMatrix();

        cudaMatrix *one, *b1;
        b1 = initCudaMatrix(nn->layers[l]->number_of_neurons, 1);
        one = initCudaMatrix(nn->minibatch_size, 1);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            (*one)[idx] = 1.0;

        matrix_dot(nn->layers[l]->delta, one, b1); // b1 <- delta^l x 1^T

        b1->copyHostToDevice();
        matrix_scalar_Kernel<<<32,32>>>(b1,  nn->alpha / nn->minibatch_size, b1); // b1 <- alpha / m . delta^l x 1^T
        // Pas de copy device to host, puisque la fonction suivante utilise b1 dans le device

        nn->layers[l]->biases->copyHostToDevice();
        matrix_minus_Kernel<<<32,32>>>(nn->layers[l]->biases, b1, nn->layers[l]->biases); // b^l = b^l - alpha / m . delta^l x 1^T
        nn->layers[l]->biases->copyDeviceToHost();

        one->destroyCudaMatrix();
        b1->destroyCudaMatrix();
    }
}