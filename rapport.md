# Table of Contents
- [Table of Contents](#table-of-contents)
- [TP Timothée - Vincent](#tp-timothée---vincent)
  - [Partie 1 : découverte du code fourni en C](#partie-1--découverte-du-code-fourni-en-c)
    - [Débugage du code en C](#débugage-du-code-en-c)

# TP Timothée - Vincent

## Partie 1 : découverte du code fourni en C

### Débugage du code en C

Apèrs avoir ouvert le code source, nous avons eu deux problèmes qui empêchent la bonne compilation :

Dans `ann.c`, la ligne suivant doit être placée en début de script pour définir correctement la constante `M_PI`.
```C
#define _USE_MATH_DEFINES // l. 1
...
```

Dans `main.c`, la valeur de `datasize` doit être constante car elle définit un array à taille fixe.
```C
...
unsigned idx[datasize]; // l. 50
...
```

Utiliser la commande suivante pour compiler :
```bash
nvcc -o tp main.cu matrix.cu ann.cu mnist.cu
```