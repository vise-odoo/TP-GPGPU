// Compile gcc -o ./ann main.c matrix.c ann.c mnist.c -lm

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "matrix.h"
#include "ann.h"
#include "cudaMatrix.h"
#include <math.h>
#include <string.h>
#include <time.h>

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define CREATE_CUDAEVENT cudaEvent_t start, stop; \
cudaEventCreate(&start); \
cudaEventCreate(&stop);
#define START_CUDAEVENT cudaEventRecord(start, 0);
#define STOP_AND_PRINT_CUDAEVENT(txt) cudaEventRecord(stop, 0);\
cudaEventSynchronize(stop);\
{float elapsedTime;\
cudaEventElapsedTime(&elapsedTime, start, stop);\
printf("Time to %s %3.8f ms\n", #txt, elapsedTime);}

#define CREATE_CPUEVENT clock_t clock_start, clock_stop;\
double cpu_time;
#define START_CPUEVENT clock_start = clock();
#define STOP_AND_PRINT_CPUEVENT(txt) clock_stop = clock();\
cpu_time = ((double) (clock_stop - clock_start)) / CLOCKS_PER_SEC;\
printf("Time to %s %3.4f s\n", #txt, cpu_time);

void populate_minibatch(double *x, double* y, unsigned* minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size, byte* label, unsigned label_size);

void zero_to_n(unsigned n, unsigned* t)
{
    for (unsigned i = 0; i < n; i++)
    {
        t[i] = i;
    }
}

void shuffle(unsigned *t, const unsigned size, const unsigned number_of_switch)
{
    zero_to_n(size, t);
    for (unsigned i = 0; i < number_of_switch; i++)
    {
        unsigned x = rand() % size;
        unsigned y = rand() % size;
        unsigned tmp = t[x];
        t[x] = t[y];
        t[y] = tmp;
    }
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double dsigmoid(double x)
{
    return sigmoid(x)*(1-sigmoid(x));
}

double accuracy(image* test_img, byte* test_label, unsigned datasize, unsigned minibatch_size, ann_t *nn)
{
    unsigned good = 0;
    unsigned idx[TEST_SIZE];   
    double *x = (double *) malloc( 28 * 28 * minibatch_size * sizeof(double));
    double *y = (double *) malloc( 10 * minibatch_size * sizeof(double) );

    zero_to_n(datasize, idx);
    for (int i = 0; i < datasize - minibatch_size; i+= minibatch_size)
    {        
        populate_minibatch(x, y, &idx[i], minibatch_size, test_img, 28*28, test_label, 10);
        memcpy(nn->layers[0]->activations->data_host, x, 28*28 * minibatch_size * sizeof(double));     
        
        forward(nn, sigmoid);
        for (int col = 0; col < minibatch_size; col ++)
        {
            int idxTrainingData = col + i ;
            double max = 0;
            unsigned idx_max = 0;
            for (int row = 0; row < 10; row++){
                int idx = col + row * minibatch_size;
                if (nn->layers[nn->number_of_layers-1]->activations->data_host[idx] > max){
                    max = nn->layers[nn->number_of_layers-1]->activations->data_host[idx];
                    idx_max = row;
                }
            }
            if (idx_max == test_label[idxTrainingData])
            {
                good ++;
            }
        }
    }    
    free(x);
    free(y);

    unsigned ntests = (datasize/minibatch_size) * minibatch_size;
    return (100.0* (double) (good) / ntests );
}

void populate_minibatch(double * x, double * y, unsigned * minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size, byte* label, unsigned label_size)
{
    for (int col = 0; col < minibatch_size; col ++)
    {
        for (int row = 0; row < img_size; row ++)
        {
            x[row * minibatch_size + col] = (double) img[minibatch_idx[col]][row]/255.;
        }

        for (int row = 0; row < 10; row ++)
        {
            y[row * minibatch_size + col] = 0.0;
        }

        y[ label[minibatch_idx[col]] * minibatch_size + col] = 1.0;
    }
}

int main(int argc, char *argv[])
{
    CREATE_CPUEVENT
    CREATE_CUDAEVENT

    START_CPUEVENT
    srand(time(0));
    unsigned datasize, ntest;
    image* train_img = read_images("train-images-idx3-ubyte", &datasize);
    byte* train_label = read_labels("train-labels-idx1-ubyte", &datasize);
    image* test_img = read_images("t10k-images-idx3-ubyte", &ntest);
    byte* test_label = read_labels("t10k-labels-idx1-ubyte", &ntest);
    STOP_AND_PRINT_CPUEVENT(load dataset)

    START_CPUEVENT
    ann_t * nn;
    double alpha = 0.05;
    unsigned minibatch_size = 16;
    unsigned number_of_layers = 3;
    unsigned nneurons_per_layer[3] = {28*28, 30, 10};
    nn = create_ann(alpha, minibatch_size, number_of_layers, nneurons_per_layer);
    //print_nn(nn);
    STOP_AND_PRINT_CPUEVENT(ANN creation)

    printf("starting accuracy %lf\n", accuracy(test_img, test_label, ntest, minibatch_size, nn));

    unsigned *shuffled_idx = (unsigned *)malloc(datasize*sizeof(unsigned));
    double *x = (double *) malloc(28*28 * minibatch_size * sizeof( double ));
    double *y = (double *) malloc(10 * minibatch_size * sizeof( double ));

    cudaMatrix* out = initCudaMatrix(10, minibatch_size);
    
    for (int epoch = 0; epoch < 40; epoch ++)
    {
        START_CPUEVENT
        printf("start learning epoch %d\n", epoch);

        shuffle(shuffled_idx, datasize, datasize);
        for (int i = 0; i < datasize - minibatch_size ; i+= minibatch_size)
        {
            populate_minibatch(x, y, shuffled_idx+i, minibatch_size, train_img, 28*28, train_label, 10);
            memcpy(nn->layers[0]->activations->data_host, x, 28 * 28 * minibatch_size * sizeof(double));
            forward(nn, sigmoid);
            memcpy(out->data_host, y, 10 * minibatch_size * sizeof(double));            
            backward(nn, out, dsigmoid);            
        }     
        printf("epoch %d accuracy %lf\n", epoch, accuracy(test_img, test_label, ntest, minibatch_size, nn));
        STOP_AND_PRINT_CPUEVENT(process one epoch)
    }

    free(x);
    free(y);
    free(shuffled_idx);
    out->destroyCudaMatrix();  

    return 0;
}

