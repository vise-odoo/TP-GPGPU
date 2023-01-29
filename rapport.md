# Table of Contents
- [Table of Contents](#table-of-contents)
- [TP Timothée MESNARD - Vincent SEVESTRE](#tp-timothée-mesnard---vincent-sevestre)
  - [Partie 1 : découverte du code fourni en C](#partie-1--découverte-du-code-fourni-en-c)
    - [Débugage du code en C](#débugage-du-code-en-c)
    - [Mise en place d'un time profiler](#mise-en-place-dun-time-profiler)
  - [Partie 2 : optimisation en utilisant le GPU](#partie-2--optimisation-en-utilisant-le-gpu)
    - [Création d'une structure adéquate aux calculs parallèles](#création-dune-structure-adéquate-aux-calculs-parallèles)
    - [Parallélisation naive des fonctions](#parallélisation-naive-des-fonctions)
    - [Parallélisation avancée](#parallélisation-avancée)

# TP Timothée MESNARD - Vincent SEVESTRE

## Partie 1 : découverte du code fourni en C

### Débugage du code en C

Apèrs avoir ouvert le code source, nous avons eu deux problèmes qui empêchent la bonne compilation (du fait que l'on soit sur Windows, utilisant le compilateur cl) :

Dans `ann.cu`, la ligne suivante doit être placée en début de script pour définir correctement la constante `M_PI`.
```C
#define _USE_MATH_DEFINES // l. 1
...
```

Dans `main.cu`, la valeur de `datasize` doit être constante car elle définit un array à taille fixe.
```C
...
unsigned idx[datasize]; // l. 50
...
```

Dans `ann.cu`, les constantes `generate` et `z1` ont été définies en constantes globales et initialisées.

```C
bool generate = false;
double z1 = 0;
```

Utiliser la commande suivante pour compiler : 
```bash
nvcc -o tp main.cu matrix.cu ann.cu mnist.cu cudaMatrix.cu
```
(Il ne faut pas oublier **`cudaMatrix.cu`**, qui est un nouveau fichier !)

### Mise en place d'un time profiler

Mise en place de `clock_t` pour déterminer ce qui prend du temps dans le processus d'apprentissage.

- Load dataset: `0.0030 s`
- Create neural network `0.0020 s`
- Process one epoch `~ 10 s`

Il faut donc optimiser la vitesse d'une epoch. Ceci est divisé en 2 partie :
- Time tu shuffle `0.0020 s`
- Boucle for : `~ 10 s`

Dans cette boucle for :
- Populate minibatch, memcpy : $\epsilon$ s
- Time to forward `~ 0.0010 s`
- Time to backward `~ 0.0010 s`

Puisque la boucle for boucle sur le nombre de données, c'est `forward` et `backward` qu'il faut optimiser.

Dans `forward`, dans une boucle for pour le nombre de couches :
- Time to initiate matrices, matrix sum, matrix function, destroy matrices :  $\epsilon$ s
- Time for matrix dot `~ 0.0010 s`

Dans `backward` :
- Time to do hadamard product :  $\epsilon$ s
- Time to do matrix transpose (inscrite dans une boucle for pour le nombre de couches) : $\epsilon$ s

On remarque que l'on est limité par la vitesse de la clock, cependant, on peut deviner la fonction chronophage : `matrix_dot`.

## Partie 2 : optimisation en utilisant le GPU

### Création d'une structure adéquate aux calculs parallèles

Dans un programme de calcul parallèle, il faut fréquemment effectuer des allocations de mémoire et des copies de variables entre le processeur et la carte graphique.
Pour simplifier le code et réduire sa taille, une structure de matrice particulière a été construite, sous la forme d'une classe `cudaMatrix` que l'on trouve dans le fichier `cudaMatrix.cu`.

En se basant sur la définition préalable de `matrix_t`, les attributs principaux de `cudaMatrix` sont :

```C++
unsigned rows; // Nombre de lignes de la matrice
unsigned columns; // Nombre de colonnes de la matrice
double* data_device; // Pointeur vers un tableau de doubles, stocké sur le CPU
double* data_host; // Pointeur vers un tableau de doubles, stocké sur le GPU
```

L'intérêt d'une telle fonction est de pouvoir créer des méthodes propres à la classe permettant d'allouer et de copier la mémoire.
Ainsi, les fonctions suivantes ont été écrites :

```C++
void allocateMemory(); // Alloue la mémoire en fonction de la taille de la matrice, i.e. rows*columns*sizeof(double)
void copyHostToDevice(); // Copie les données de data_host vers data_device
void copyDeviceToHost(); // Copie les données de data_device vers data_host
void destroyCudaMatrix(); // Supprime les tableaux data_host et data_device
```

Dans les fonctions parallélisées, on pourra donc se passer d'appeler `cudaMalloc` et `cudaMemcpy` en utilisant uniquement les méthodes de la classe.

Pour initialiser une `cudaMatrix`, l'appel est le suivant :

```C++
cudaMatrix* m = initCudaMatrix(rows, columns);
```

La fonction `initCudaMatrix` renvoie un pointeur vers une `cudaMatrix ` de taille rows*columns, où les allocations de mémoire CPU & GPU ont déjà été effectuées.
### Parallélisation naive des fonctions

Pour commencer l'amélioration du programme, une implémentation naive des fonctions matricielles a été réalisée. Ces fonctions peuvent être consultées dans le fichier `matrix.cu`.
Cette implémentation naive a pour but de retirer la plupart des boucles `for` en utilisant une grille de threads, qui peuvent être aussi bien en 1D qu'en 2D.

Chacune de ces fonctions parallélisées est en deux parties : 

- La première partie est de la forme `func_Kernel(cudaMatrix* m, ...)`, c'est ce type de fonction qui est appelé dans `main.cu` et `ann.cu`. Ces fonctions réalisent les instructions d'assertion, puis envoient le calcul sur le GPU pour le réaliser.

- La seconde partie est de la forme `__global__ func_Device(double* m, ...)`. Ces fonctions effectuent le calcul sur le GPU.

Donnons un exemple d'utilisation de ces fonctions :

```C++
void func_Kernel(cudaMatrix* m1, cudaMatrix* res){
  assert((m1->rows == res->rows) && 
         (m1->columns == res-> columns))

  m1->copyHostToDevice();
  func_Device<<<N_BLOCK, DIM_BLOCK>>>(m1->data_device, res->data_device, m1->rows, m1->columns);
  res->copyDeviceToHost();
}
```
```C++
__global__ func_Device(double* m1, double* res, int rows, int cols){
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < rows * cols){res[idx] = func(m1[idx])}
}
```

Toutes les fonctions du fichier `matrix.cu` ont subi le traitement de ce parallélisme. Ainsi, on peut décliner l'ajout de `matrix_sum_Kernel, matrix_minus_Kernel, matrix_scalar_Kernel, matrix_function_Kernel, hadamard_product_Kernel, matrix_dot_Kernel` et de `matrix_transpose_Kernel`.

Le gain de temps est le suivant : 

- En utilisant la structure `cudaMatrix` sur CPU, sans utiliser les fonctions parallélisées : 17.8 s/epoch, soit une perte de temps de 7.8 s/epoch.
- Avec la structure `cudaMatrix` et les fonctions élementaires sur GPU (`matrix_sum_Kernel, matrix_minus_Kernel, matrix_scalar_Kernel, matrix_function_Kernel, hadamard_product_Kernel`) : 18 s/epoch, soit une perte de temps de 8 s/epoch.
- Avec la structure `cudaMatrix` et toutes les fonctions sur le GPU, 6 s/epoch, soit un gain de temps de 4 s/epoch.

### Parallélisation avancée

L'implémentation naive peut évidemment être améliorée. Par exemple, les mécanismes de **mémoire partagée** et d'**accès fusionné** n'ont pas été pris en compte dans l'écriture des fonctions précédentes.

Les fonctions occupant le plus de mémoire et prenant le plus de temps de calcul vont être raffinées.

- Pour `matrix_transpose`, une mémoire partagée est créée pour importer dans un tableau temporaire les valeurs locales de la matrice à transposer. Ainsi, un accès fusionné aux valeurs est effectué. Les chargements mémoires sont limités, et le temps d'exécution est amélioré.

```C++
__global__ void matrix_transpose_shared_Device(double* m1, double* res, int rows, int cols)
{
	__shared__ float shared[32][32];
	
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if((xIndex < rows) && (yIndex < cols)){shared[threadIdx.y][threadIdx.x] = m1[yIndex * rows + xIndex];} // Copie dans la mémoire partagée

	__syncthreads();

	if((xIndex < cols) && (yIndex < rows)){res[yIndex * cols + xIndex;] = shared[threadIdx.x][threadIdx.y];} // Copie dans le résultat
}
```

Le gain de temps est le suivant : 
- Avec `matrix_transpose_shared`, le gain de temps est de **#TODO** s, soit un gain de temps de **#TODO** s.