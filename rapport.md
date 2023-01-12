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

- Load dataset: `... s`
- Create neural network `... s`
- 

## Partie 2 : optimisation en utilisant le GPU