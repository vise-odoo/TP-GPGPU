# Table of Contents
- [Table of Contents](#table-of-contents)
- [TP Timothée - Vincent](#tp-timothée---vincent)
  - [Partie 1 : découverte du code fourni en C](#partie-1--découverte-du-code-fourni-en-c)
    - [Débugage du code en C](#débugage-du-code-en-c)
    - [Mise en place d'un time profiler](#mise-en-place-dun-time-profiler)
  - [Partie 2 : optimisation en utilisant le GPU](#partie-2--optimisation-en-utilisant-le-gpu)

# TP Timothée - Vincent

## Partie 1 : découverte du code fourni en C

### Débugage du code en C

Apèrs avoir ouvert le code source, nous avons eu deux problèmes qui empêchent la bonne compilation (du fait que l'on soit sur Windows, utilisant le compilateur cl) :

Dans `ann.cu`, la ligne suivant doit être placée en début de script pour définir correctement la constante `M_PI`.
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
nvcc -o tp main.cu matrix.cu ann.cu mnist.cu
```

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

